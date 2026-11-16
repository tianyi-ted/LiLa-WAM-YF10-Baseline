"""Draw the implemented D-uniform + inference-only action-tail feedback.

Only writes illustration assets; never loads or changes model weights/data.
SVG text remains editable. PNG/PDF are rendered from the same vector objects.
Run from any directory: python utils/figures/draw_d_uniform_tail.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
W, H = 2400, 1540
C = dict(ink="#24374b", muted="#64758a", line="#50647d", blue="#377db7",
         bluefill="#e8f2fb", orange="#b77b2c", orangefill="#fff4df",
         purple="#8765bd", purplefill="#f0e9fa", green="#35866e",
         greenfill="#e5f4ee", rose="#b85c72", rosefill="#fbeef2",
         grey="#8d99a7", greyfill="#f0f3f6", white="#ffffff")


class Diagram:
    def __init__(self, language):
        self.zh = language == "zh"
        self.language = language
        self.fig = plt.figure(figsize=(W / 100, H / 100), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
        self.ax.axis("off")
        cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        self.font = font_manager.FontProperties(fname=str(cjk)) if self.zh else font_manager.FontProperties(family="DejaVu Sans")
        self.text_checks = []

    def tr(self, zh, en):
        return zh if self.zh else en

    def text(self, x, y, label, size=20, color=None, bold=False, ha="center", maxwidth=None):
        obj = self.ax.text(x, y, label, fontsize=size * .72, fontproperties=self.font,
                           fontweight="bold" if bold else "normal", color=color or C["ink"],
                           ha=ha, va="center", linespacing=1.5, zorder=10)
        if maxwidth is not None:
            self.text_checks.append((obj, maxwidth))
        return obj

    def rect(self, x, y, w, h, fill, edge=None, radius=12, dash=False, lw=1.5):
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge or fill, linewidth=lw,
                              linestyle=(0, (6, 4)) if dash else "solid", zorder=1)
        self.ax.add_patch(patch)
        return patch

    def box(self, x, y, w, h, title, detail="", fill="bluefill", edge="blue", size=21):
        self.rect(x, y, w, h, C[fill], C[edge])
        self.text(x + w / 2, y + h * (.36 if detail else .5), title, size, bold=True, maxwidth=w-22)
        if detail:
            self.text(x + w / 2, y + h * .72, detail, 16, C["muted"], maxwidth=w-22)

    def arrow(self, points, color="line", dash=False, width=1.8, head=True):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-1))
        p = FancyArrowPatch(path=path, arrowstyle="-|>" if head else "-",
                            mutation_scale=14, linewidth=width, color=C.get(color, color),
                            linestyle=(0, (5, 4)) if dash else "solid", zorder=4,
                            capstyle="round", joinstyle="round")
        self.ax.add_patch(p)

    def panel(self, y, h, title, note, color):
        self.rect(45, y, 2310, h, "#fcfdff", C[color], dash=True)
        self.text(72, y+29, title, 25, C[color], bold=True, ha="left")
        self.text(2327, y+29, note, 17, C["muted"], ha="right")

    def frame(self, x, y, label, future=False, width=155, height=66):
        col = "orange" if future else "blue"
        self.rect(x, y, width, height, "#fafbfc", C[col], radius=5)
        # Diagrammatic robot/camera glyph, not a simulated observation.
        self.ax.add_patch(Rectangle((x+12, y+height-19), 54, 5, facecolor="#c0cbd5", zorder=2))
        pts=[(x+23, y+height-20), (x+27, y+25), (x+46, y+17), (x+55, y+32)]
        self.ax.plot(*zip(*pts), color=C[col], linewidth=3, zorder=3)
        for a,b in pts[:-1]:
            self.ax.add_patch(Circle((a,b), 3, facecolor="white", edgecolor=C[col], zorder=4))
        self.ax.add_patch(Rectangle((x+64, y+height-32), 11, 12, facecolor=C["green"], zorder=3))
        self.text(x+width*.72, y+height/2, label, 17, maxwidth=width*.48)

    def latent(self, x, y, teacher=False):
        self.rect(x, y, 100, 54, C["purplefill"], C["purple"])
        self.text(x+50, y+18, "zT" if teacher else "zS", 22, C["purple"], True)
        self.text(x+50, y+40, "1 × 768", 14, C["muted"])

    def expert(self, x, y, w=520, teacher=False, inference=False):
        self.rect(x, y, w, 126, C["greenfill"], C["green"])
        self.ax.add_patch(Rectangle((x+1,y+1), w*.31, 124, facecolor=C["bluefill"], edgecolor="none", zorder=2))
        self.text(x+w*.155, y+36, self.tr("融合 / 适配器", "Fusion / adapter"), 19, maxwidth=w*.3-12)
        self.text(x+w*.155, y+76, self.tr("多层当前特征", "Current features"), 15, maxwidth=w*.3-12)
        title = self.tr("动作 DiT", "Action DiT")
        self.text(x+w*.66, y+30, title, 24, bold=True)
        self.text(x+w*.66, y+60, self.tr("12 层 · h512 · 无 gate", "12 blocks · h512 · no gate"), 17, maxwidth=w*.66-15)
        labels=[self.tr("状态", "State"),self.tr("意图", "Intent"),self.tr("噪声动作", "Noisy action")]
        colors=["orange","purple","green"]
        start=x+w*.34
        for i,(label,col) in enumerate(zip(labels,colors)):
            xx=start+i*w*.205
            self.rect(xx,y+83,w*.185,24,C[col+"fill"],C[col],radius=5,lw=1)
            self.text(xx+w*.0925,y+95,label,13,maxwidth=w*.185-7)

    def draw(self):
        t=self.tr
        self.text(65,45,t("D-uniform ＋ 动作尾部回传：训练与部署通路", "D-uniform with Action-tail Feedback"),35,bold=True,ha="left")
        self.text(65,88,t("冻结 DINOv3-B  ·  因果 task-attention 意图  ·  latent KD  ·  推理时尾部复用", "Frozen DINOv3-B  /  Causal task-attention intent  /  Latent KD  /  Inference-only tail reuse"),21,C["muted"],ha="left")
        self.arrow([(70,132),(135,132)],"blue")
        self.text(148,132,t("当前观测 / 特征", "Current inputs / features"),17,ha="left")
        self.arrow([(505,132),(570,132)],"orange")
        self.text(583,132,t("真实未来（仅训练）", "Real future: training only"),17,ha="left")
        self.arrow([(1010,132),(1075,132)],"purple",dash=True)
        self.text(1088,132,t("latent 监督 / 停止梯度", "Latent supervision / stop-gradient"),17,ha="left")
        self.arrow([(1640,132),(1705,132)],"grey",dash=True)
        self.text(1718,132,t("一次性兼容参数初始化", "One-time compatible initialization"),17,ha="left")

        # A: separately trained privileged teacher. No student noise/action enters posterior.
        self.panel(178,315,t("A  教师阶段：用真实未来塑造动作相关 latent", "A  Teacher stage: action-shaped privileged latent"),t("posterior 与动作网络共同训练；DINO 冻结", "Train posterior + action network; freeze DINO"),"orange")
        self.frame(85,300,"I(t)")
        self.frame(85,390,"I(t+8)",future=True)
        self.box(300,334,180,105,"DINOv3-B",t("冻结 · 共享权重", "Frozen · shared"))
        self.arrow([(240,333),(270,333),(270,365),(300,365)],"blue")
        self.arrow([(240,423),(270,423),(270,405),(300,405)],"orange")
        self.box(710,310,340,125,t("未来驱动 posterior", "Future-driven posterior"),t("未来 patch 查询当前上下文\n残差交互 → learned summary", "Future queries read current context\nResidual interaction → summary"),"orangefill","orange")
        self.arrow([(480,365),(710,350)],"blue")
        self.arrow([(480,405),(710,400)],"orange")
        self.text(589,341,"F(t)",17,C["blue"])
        self.text(590,424,"F(t+8)",17,C["orange"])
        self.box(710,450,340,29,t("状态 s(t) ＋ VTT ＋ H=8", "State s(t) + VTT + H=8"),fill="greyfill",edge="grey",size=16)
        self.arrow([(880,450),(880,435)],"line")
        self.latent(1125,345,True)
        self.arrow([(1050,372),(1125,372)],"orange")
        self.arrow([(1225,372),(1320,372)],"purple")
        self.text(1272,343,t("LN＋投影", "LN + proj."),14,C["purple"])
        self.expert(1320,310,teacher=True)
        self.arrow([(480,355),(535,355),(535,270),(1400,270),(1400,310)],"blue")
        self.text(1010,251,t("当前多层 DINO 特征", "Current multi-layer DINO features"),16,C["blue"])
        self.box(1320,450,520,29,"xτ = (1−τ) ε + τ a     |     τ, s(t)",fill="greyfill",edge="grey",size=16)
        self.arrow([(1580,450),(1580,436)],"line")
        self.box(2030,338,260,85,t("专家动作监督", "Expert action loss"),"LFM = MSE(vT, a − ε)","rosefill","rose")
        self.arrow([(1840,372),(2030,372)],"green")
        self.text(1934,347,"vT",20,C["green"])
        self.text(2160,455,t("教师不预测 / 重建 RGB", "No RGB reconstruction"),17,C["muted"])

        # B: teacher posterior only is retained as a frozen target generator.
        self.panel(525,440,t("B  学生阶段：兼容初始化 ＋ latent 蒸馏", "B  Student stage: compatible initialization + latent KD"),t("D-uniform：统一 LR 2e−4 → 5e−5；预测器重新初始化", "D-uniform: uniform LR 2e−4 → 5e−5; fresh predictor"),"purple")
        self.frame(85,700,"I(t)")
        self.frame(85,800,"I(t+8)",future=True)
        self.box(300,727,180,105,"DINOv3-B",t("冻结 · 共享权重", "Frozen · shared"))
        self.arrow([(240,733),(270,733),(270,755),(300,755)],"blue")
        self.arrow([(240,833),(270,833),(270,805),(300,805)],"orange")
        self.box(710,615,340,90,t("冻结教师 posterior", "Frozen teacher posterior"),t("当前＋真实未来＋状态＋VTT", "Current + future + state + VTT"),"orangefill","orange")
        self.arrow([(480,800),(560,800),(560,675),(710,675)],"orange")
        self.arrow([(480,757),(590,757),(590,642),(710,642)],"blue")
        self.text(1083,638,"zT",19,C["purple"])
        self.arrow([(1050,660),(1130,660)],"purple",dash=True)
        self.box(1130,615,300,90,t("latent 蒸馏", "Latent distillation"),"LKD = MSE(LN(zS), sg LN(zT))", "purplefill","purple",size=22)
        self.box(1960,615,330,90,t("学生总损失", "Student objective"),"L = LFM + 0.1 LKD","rosefill","rose")
        self.arrow([(1430,660),(1960,660)],"purple",dash=True)
        self.box(710,770,340,120,t("因果 G2 预测器", "Causal G2 predictor"),t("task/state query → patch attention\n单向量汇总 → 条件 MLP", "Task/state query → patch attention\nSingle-vector pool → conditional MLP"),"purplefill","purple")
        self.arrow([(480,757),(620,757),(620,830),(710,830)],"blue")
        self.latent(1125,803)
        self.arrow([(1050,830),(1125,830)],"purple")
        self.arrow([(1175,803),(1175,742),(1280,742),(1280,705)],"purple",dash=True)
        self.expert(1320,770)
        self.arrow([(1225,830),(1320,830)],"purple")
        self.text(1272,800,t("LN＋投影", "LN + proj."),14,C["purple"])
        self.arrow([(480,750),(525,750),(525,590),(1455,590),(1455,770)],"blue")
        self.text(895,603,t("当前多层特征 → 学生视觉适配器", "Current multi-layer features → student visual adapter"),16,C["blue"])
        self.box(2030,790,260,85,t("专家动作监督", "Expert action loss"),"LFM = MSE(vS, a − ε)","rosefill","rose")
        self.arrow([(1840,830),(2030,830)],"green")
        self.text(1930,803,"vS",20,C["green"])
        self.arrow([(2160,790),(2160,705)],"rose",dash=True)
        self.box(710,917,340,29,t("状态 s(t) ＋ VTT ＋ H=8", "State s(t) + VTT + H=8"),fill="greyfill",edge="grey",size=16)
        self.arrow([(880,917),(880,890)])
        self.box(1320,917,520,29,"xτ = (1−τ) ε + τ a     |     τ, s(t)",fill="greyfill",edge="grey",size=16)
        self.arrow([(1580,917),(1580,896)])
        # Source transfer includes adapters/projections/heads, not just DiT.
        self.arrow([(1750,436),(1750,510),(1880,510),(1880,740),(1750,740),(1750,770)],"grey",dash=True)
        self.text(1910,509,t("教师 epoch40 → 兼容动作模块", "Teacher ep40 → compatible action modules"),16,C["muted"],ha="left",maxwidth=420)
        self.text(1960,925,t("KD 不直接更新学生 DiT；动作 loss 更新整条学生通路", "KD: predictor only; action loss: whole student path"),15,C["muted"],ha="left",maxwidth=350)

        # C: online flow inference, no teacher/future; feedback modifies x0, not gradient.
        self.panel(998,493,t("C  在线推理：因果意图 ＋ 未执行动作尾部回传", "C  Online inference: causal intent + unexecuted action-tail feedback"),t("无教师 / 真实未来 / 蒸馏 loss；无梯度回传", "No teacher, future input, KD or backpropagation"),"green")
        self.frame(85,1142,"I(t)",height=82)
        self.box(300,1130,180,100,"DINOv3-B",t("冻结", "Frozen"))
        self.arrow([(240,1183),(300,1183)],"blue")
        self.box(680,1130,310,100,t("因果 task-attention G2", "Causal task-attention G2"),t("当前 patch → zS", "Current patches → zS"),"purplefill","purple",size=20)
        self.arrow([(480,1180),(680,1180)],"blue")
        self.box(1060,1130,180,100,t("意图 token", "Intent token"),"LN + Linear\n768 → 512","purplefill","purple",size=21)
        self.arrow([(990,1180),(1060,1180)],"purple")
        self.box(1340,1130,350,100,t("学生动作网络 ＋ Euler", "Student action net + Euler"),t("当前视觉 / 状态 / 意图\n10 次速度场积分", "Current vision / state / intent\n10 velocity-field evaluations"),"greenfill","green",size=21)
        self.arrow([(1240,1180),(1340,1180)],"purple")
        self.arrow([(480,1150),(565,1150),(565,1092),(1450,1092),(1450,1130)],"blue")
        self.text(1020,1075,t("当前多层视觉特征（不经过意图池化）", "Current multi-layer vision bypasses intent pooling"),16,C["blue"])
        self.rect(1750,1125,260,145,C["greyfill"],C["grey"])
        self.text(1880,1150,t("归一化动作块 8 × 4", "Normalized action chunk 8 × 4"),19,bold=True,maxwidth=240)
        for i in range(8):
            col="blue" if i<4 else "green"
            self.rect(1764+i*29,1180,24,30,C[col+"fill"],C[col],radius=4,lw=1.2)
        self.text(1818,1233,t("执行前4步", "Execute 4"),16,C["blue"])
        self.text(1940,1233,t("保留后4步", "Retain 4"),16,C["green"])
        self.arrow([(1690,1180),(1750,1180)],"green")
        self.box(2090,1130,220,100,t("环境执行", "Environment"),t("反归一化 / 裁剪\n每步发送一个动作", "Denormalize / clip\nSend one action per step"),"bluefill","blue",size=23)
        self.arrow([(2010,1180),(2090,1180)],"blue")
        self.box(680,1260,310,52,t("状态 ＋ VTT ＋ H=8", "State + VTT + H=8"),fill="greyfill",edge="grey",size=18)
        self.arrow([(835,1260),(835,1230)])
        self.text(1148,1272,t("无直接 VTT token；无 gate", "No direct VTT token; no gate"),16,C["muted"])
        self.box(1520,1255,170,36,t("归一化状态 s(t)", "Normalized state s(t)"),fill="greyfill",edge="grey",size=16)
        self.arrow([(1605,1255),(1605,1230)])
        self.box(1720,1330,305,100,t("归一化动作缓存", "Normalized action cache"),t("去掉已执行4步\n取未执行4步 tail", "Drop the 4 executed actions\nKeep 4 unexecuted actions"),"greenfill","green",size=21)
        self.arrow([(1880,1270),(1880,1330)],"green")
        self.box(1040,1330,565,100,t("下一轮 Flow 初始状态 x0", "Next-cycle Flow initial state x0"),"x0[0:4] = 1.0 × tail + 0.5 × ε[0:4]\nx0[4:8] = ε[4:8]","greenfill","green",size=21)
        self.arrow([(1720,1380),(1605,1380)],"green",width=2.4)
        self.box(770,1352,175,56,"ε ~ N(0, I)",fill="greyfill",edge="grey",size=21)
        self.arrow([(945,1380),(1040,1380)],"line")
        self.arrow([(1320,1330),(1320,1302),(1435,1302),(1435,1230)],"green",width=2.4)
        self.text(1528,1299,t("下轮规划使用", "Next planning cycle"),16,C["green"])
        self.text(380,1345,t("首个动作块：全部使用标准高斯噪声\n后续每执行4步，刷新观测并重新规划", "First chunk: standard Gaussian noise only\nRe-observe and replan after every 4 actions"),18,C["muted"],maxwidth=620)
        self.arrow([(2200,1230),(2200,1460),(160,1460),(160,1224)],"grey",dash=True)
        self.rect(650,1444,1115,30,"#fcfdff",radius=0).set_zorder(8)
        self.text(1207,1459,t("环境状态演化 → 新观测（不是反向传播）", "Environment evolves → new observation (not backpropagation)"),17,C["muted"])
        self.text(65,1520,t("范围：Meta-World MT50 / corner2；C8 / E4 / H8。训练使用高斯 FM；尾部回传只在推理开启。图示 patch / token 为示意，非实际数量。", "Scope: Meta-World MT50 / corner2; C8 / E4 / H8. Gaussian FM training; tail feedback at inference only. Patch/token glyphs are schematic."),17,C["muted"],ha="left",maxwidth=2270)
        return self

    def save(self, out, dpi):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        # Fit long translated labels to their allocated columns without clipping.
        for obj,width in self.text_checks:
            measured=obj.get_window_extent(renderer=renderer).width
            if measured>width:
                obj.set_fontsize(obj.get_fontsize()*width/measured*.98)
        self.fig.canvas.draw()
        issues=[]
        for obj,width in self.text_checks:
            bb=obj.get_window_extent(renderer=self.fig.canvas.get_renderer())
            if bb.width>width+2:
                issues.append(obj.get_text())
        if issues:
            raise RuntimeError(f"Text overflow: {issues}")
        stem=out/f"d_uniform_tail_overview_{self.language}"
        for ext in ["svg","pdf","png"]:
            self.fig.savefig(stem.with_suffix('.'+ext),dpi=dpi,facecolor="white")
        plt.close(self.fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--dpi',type=int,default=200)
    args=parser.parse_args()
    if args.dpi<100:
        parser.error('dpi must be at least 100')
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for lang in ['zh','en']:
        Diagram(lang).draw().save(args.output_dir,args.dpi)
    print(json.dumps({'output':str(args.output_dir),'languages':['zh','en'],
                      'formats':['svg','pdf','png'],'raster_size':[W*args.dpi//100,H*args.dpi//100]},ensure_ascii=False))


if __name__=='__main__':
    main()
