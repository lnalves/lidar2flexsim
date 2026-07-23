"""Gera imagem da segmentação (vista 3D + vista superior) com matplotlib."""
import numpy as np, open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud("cena_demo.ply")
pcd = pcd.voxel_down_sample(0.03)
pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
_, inl = pcd.segment_plane(0.03, 3, 1000)
chao = np.asarray(pcd.select_by_index(inl).points)
obj = pcd.select_by_index(inl, invert=True)
labels = np.array(obj.cluster_dbscan(eps=0.22, min_points=60))
pts = np.asarray(obj.points)
cores = plt.cm.tab10(np.linspace(0, 1, 10))
nomes = {0: "operador", 1: "operador", 2: "bancada", 3: "esteira"}

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection="3d")
ax.scatter(chao[::8,0], chao[::8,1], chao[::8,2], s=0.3, c="lightgray", label="chão")
for i in range(labels.max()+1):
    m = labels == i
    ax.scatter(pts[m,0], pts[m,1], pts[m,2], s=1.2, color=cores[i],
               label=f"{nomes.get(i,'?')} #{i+1}")
ax.set_title("Segmentação 3D (RANSAC + DBSCAN)")
ax.set_box_aspect((10, 8, 3)); ax.legend(markerscale=8, fontsize=8)

ax2 = fig.add_subplot(122)
ax2.scatter(chao[::4,0], chao[::4,1], s=0.3, c="lightgray")
for i in range(labels.max()+1):
    m = labels == i
    ax2.scatter(pts[m,0], pts[m,1], s=1.5, color=cores[i])
    c = pts[m].mean(0)
    ax2.annotate(nomes.get(i,"?"), (c[0], c[1]), fontsize=10, weight="bold")
ax2.set_title("Vista superior (layout p/ FlexSim)")
ax2.set_aspect("equal"); ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
plt.tight_layout(); plt.savefig("saida/segmentacao.png", dpi=130)
print("ok")
