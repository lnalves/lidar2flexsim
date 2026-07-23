"""
Gera uma nuvem de pontos sintética simulando um scan LiDAR de um ambiente
industrial: chão + 1 esteira + 1 bancada + 2 operadores, com ruído gaussiano
típico de sensor. Serve para testar o pipeline sem hardware LiDAR.

Uso: python gerar_cena_demo.py  →  gera cena_demo.ply
"""

import numpy as np
import open3d as o3d

rng = np.random.default_rng(42)
RUIDO = 0.008  


def amostrar_caixa(centro, dim, n, apenas_superficie=True):
    cx, cy, cz = centro
    dx, dy, dz = dim
    pontos = []

    faces = [
        ("topo", n // 3), ("frente", n // 6), ("tras", n // 6),
        ("esq", n // 6), ("dir", n // 6),
    ]
    for face, m in faces:
        u = rng.uniform(-0.5, 0.5, m)
        v = rng.uniform(-0.5, 0.5, m)
        if face == "topo":
            p = np.c_[u * dx, v * dy, np.full(m, dz / 2)]
        elif face == "frente":
            p = np.c_[u * dx, np.full(m, -dy / 2), v * dz]
        elif face == "tras":
            p = np.c_[u * dx, np.full(m, dy / 2), v * dz]
        elif face == "esq":
            p = np.c_[np.full(m, -dx / 2), u * dy, v * dz]
        else:
            p = np.c_[np.full(m, dx / 2), u * dy, v * dz]
        pontos.append(p + [cx, cy, cz])
    return np.vstack(pontos)


def amostrar_operador(pos, n=1500):

    x, y = pos
    m = int(n * 0.8)
    theta = rng.uniform(0, 2 * np.pi, m)
    r = 0.22 * np.sqrt(rng.uniform(0.7, 1.0, m))
    z = rng.uniform(0.0, 1.45, m)
    corpo = np.c_[x + r * np.cos(theta), y + r * np.sin(theta), z]
    k = n - m
    phi = rng.uniform(0, np.pi, k)
    theta2 = rng.uniform(0, 2 * np.pi, k)
    cabeca = np.c_[x + 0.11 * np.sin(phi) * np.cos(theta2),
                   y + 0.11 * np.sin(phi) * np.sin(theta2),
                   1.58 + 0.11 * np.cos(phi)]
    return np.vstack([corpo, cabeca])

partes = []


gx, gy = np.meshgrid(np.linspace(-5, 5, 220), np.linspace(-4, 4, 180))
chao = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
partes.append(chao)

partes.append(amostrar_caixa((0.0, 1.5, 0.45), (4.0, 0.6, 0.9), 6000))

partes.append(amostrar_caixa((2.5, -1.5, 0.45), (1.5, 0.8, 0.9), 3000))


partes.append(amostrar_operador((-2.0, -1.0)))
partes.append(amostrar_operador((1.2, -2.5)))

nuvem = np.vstack(partes)
nuvem += rng.normal(0, RUIDO, nuvem.shape) 

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(nuvem)
o3d.io.write_point_cloud("cena_demo.ply", pcd)
print(f"Cena sintética gerada: cena_demo.ply ({len(nuvem):,} pontos)")
print("Conteúdo: chão + 1 esteira + 1 bancada + 2 operadores")
