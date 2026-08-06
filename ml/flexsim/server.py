"""Servidor HTTP que publica a cena mais recente para o FlexSim.

É o caminho de menor latência da ponte: o pipeline atualiza um único valor
em memória e o FlexSim (ou qualquer outro cliente) lê o estado atual quando
quiser, sem tocar no disco. A escrita em arquivo continua existindo em
paralelo, para instalações sem rede entre as duas máquinas.

O bind padrão é ``127.0.0.1``: enquanto o FlexSim roda na mesma máquina, a
cena não fica exposta à rede do laboratório. Para o caso de máquinas
separadas, ``--serve-host 0.0.0.0`` é explícito e consciente.
"""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from .scene import empty_scene, scene_to_csv, scene_to_json


class SceneStore:
    """Guarda a última cena e as suas serializações, pronto para leitura."""

    def __init__(self, scene: Mapping[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._scene: dict[str, Any] = dict(scene or empty_scene())
        self._json = scene_to_json(self._scene)
        self._csv = scene_to_csv(self._scene)
        self._updated_at = time.time()
        self._revision = 0

    def publish(self, scene: Mapping[str, Any]) -> None:
        payload = dict(scene)
        # Serializa fora do lock apenas o que é caro? Não: as duas formas são
        # pequenas e serializar aqui garante que json e csv nunca representem
        # cenas diferentes para dois leitores simultâneos.
        encoded_json = scene_to_json(payload)
        encoded_csv = scene_to_csv(payload)
        with self._lock:
            self._scene = payload
            self._json = encoded_json
            self._csv = encoded_csv
            self._updated_at = time.time()
            self._revision += 1

    def snapshot(self) -> tuple[dict[str, Any], str, str, float, int]:
        with self._lock:
            return (
                dict(self._scene),
                self._json,
                self._csv,
                self._updated_at,
                self._revision,
            )

    @property
    def scene(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._scene)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def health(self) -> dict[str, Any]:
        scene, _, _, updated_at, revision = self.snapshot()
        return {
            "status": "ok",
            "revision": revision,
            "updated_at": updated_at,
            "age_seconds": round(max(0.0, time.time() - updated_at), 3),
            "frame": scene.get("frame"),
            "objects": len(scene.get("objects") or []),
            "format": scene.get("format"),
        }


class _SceneRequestHandler(BaseHTTPRequestHandler):
    server_version = "lidar2flexsim/1.0"
    store: SceneStore

    def do_GET(self) -> None:  # noqa: N802 - assinatura da stdlib
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route in {"/scene.csv", "/csv"}:
            _, _, csv_text, _, _ = self.store.snapshot()
            self._respond(HTTPStatus.OK, csv_text, "text/csv; charset=utf-8")
        elif route in {"/scene", "/scene.json", "/"}:
            _, json_text, _, _, _ = self.store.snapshot()
            self._respond(HTTPStatus.OK, json_text, "application/json; charset=utf-8")
        elif route == "/health":
            payload = json.dumps(self.store.health(), ensure_ascii=False)
            self._respond(HTTPStatus.OK, payload, "application/json; charset=utf-8")
        else:
            payload = json.dumps({"error": "rota desconhecida", "path": route})
            self._respond(HTTPStatus.NOT_FOUND, payload, "application/json; charset=utf-8")

    def _respond(self, status: HTTPStatus, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        # O FlexSim consulta em laço; qualquer cache intermediário devolveria
        # um armazém desatualizado sem nenhum sinal de que está velho.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - assinatura da stdlib
        """Silencia o log por requisição: são dez por segundo."""


class SceneServer:
    """Servidor HTTP em thread própria, ligado a um :class:`SceneStore`."""

    def __init__(
        self,
        store: SceneStore | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.store = store or SceneStore()
        self.host = str(host)
        self.requested_port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self.requested_port

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in {"", "0.0.0.0"} else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> "SceneServer":
        if self._server is not None:
            raise RuntimeError("O servidor de cena já está em execução.")
        handler = type(
            "BoundSceneRequestHandler", (_SceneRequestHandler,), {"store": self.store}
        )
        self._server = ThreadingHTTPServer((self.host, self.requested_port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="flexsim-scene-server",
            daemon=True,
        )
        self._thread.start()
        return self

    def publish(self, scene: Mapping[str, Any]) -> None:
        self.store.publish(scene)

    def stop(self, timeout: float = 5.0) -> None:
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=timeout)

    def __enter__(self) -> "SceneServer":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


__all__ = ["SceneServer", "SceneStore"]
