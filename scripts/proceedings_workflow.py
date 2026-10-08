"""Publication adaptation of the author's preserved workflow attachment.

The topology follows the original: patient sets and singleton pools -> two-stage
sampling -> binary matrix/GNN responses -> conditional and marginal explanations.
Historical classifier/CV/physician claims are explicitly distinguished.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from proceedings_style import COLORS, configure, save


def workflow(output):
    configure()
    fig, ax = plt.subplots(figsize=(5.5, 6.2))
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    ax.set(xlim=(0, 10), ylim=(0, 12)); ax.axis("off")
    def box(x,y,w,h,text,color="#777777",fontsize=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.06,rounding_size=.08",
                                   facecolor="white",edgecolor=color,linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize,linespacing=1.3)
    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=8,
                                    linewidth=.8,color="#555555"))
    def heading(y,letter,text):
        ax.text(.1,y,letter,fontsize=11,weight="bold",va="top")
        ax.text(.6,y,text,fontsize=9,weight="bold",va="top")
    heading(11.9,"A","Patient image sets and model-predicted image pools")
    box(.2,10.35,2.4,.9,"One patient's\nultrasound image set")
    box(3.2,10.35,2.9,.9,"Grouped-image GNN\nDenseNet121 + GAT")
    box(6.7,10.35,3,.9,"Full-set class-1\nprobability")
    arrow(2.65,10.8,3.1,10.8);arrow(6.15,10.8,6.6,10.8)
    box(.2,9.0,2.4,.85,"Each image alone\n(singleton inference)")
    box(3.2,9.0,2.9,.85,"Same grouped-image\nGNN / singleton graph")
    box(6.7,9.0,3,.85,"Predicted class 0 / 1 pools\nnot observed image labels",fontsize=7.7)
    arrow(1.4,10.3,1.4,9.9);arrow(2.65,9.43,3.1,9.43);arrow(6.15,9.43,6.6,9.43)
    heading(8.55,"B","Perturbation distributions: 1,000 training rows per arm")
    box(.2,6.1,4.1,1.85,"Random arm\nSubset size uniform from 3 to n\nImages sampled without replacement\nDuplicates retained as fresh queries",COLORS['random'])
    box(4.9,6.1,4.8,1.85,"Adaptive arm\nI: random until a class quota is reached\nII: intended 85/15 pool-biased draws\nPool deficits reallocated; one-pool fallback",COLORS['adaptive'])
    ax.text(7.3,5.65,"Requested 50/50 predictions;\nattainment is measured.",fontsize=7.5,ha="center",linespacing=1.2)
    arrow(8.2,8.95,8.2,8.05)
    heading(5.25,"C","Binary design matrix and matched GNN responses")
    box(.2,3.6,4.1,1.05,"U: rows = sampled subsets\ncolumns = image inclusion (0 / 1)")
    box(4.9,3.6,4.8,1.05,"For every row: rebuild the graph\nGNN response = class-1 probability")
    arrow(2.2,6.05,2.2,4.7);arrow(7.3,5.55,7.3,4.7);arrow(4.35,4.1,4.85,4.1)
    heading(3.15,"D","Conditional and marginal model-output associations")
    box(.2,1.7,3,.9,"Fixed-alpha Ridge\nprobability regression",COLORS['ridge'])
    box(3.5,1.7,3,.9,"Elastic Net regression\ntraining-only 5-fold CV",COLORS['elastic_net'])
    box(6.8,1.7,2.9,.9,"Pearson inclusion–\nprobability correlation",COLORS['pearson'])
    arrow(1.7,2.95,1.7,2.65);arrow(5.0,2.95,5.0,2.65);arrow(8.2,2.95,8.2,2.65)
    ax.text(.2,1.12,"Evaluate: shared-novel fidelity, seed stability and node-deletion controls.",fontsize=8)
    ax.text(.2,.55,"Adapted from the author's original workflow. Historical classifier/10-fold CV,\nbootstrap inference and physician evaluation are not completed analyses here.",
            fontsize=7.5,va="center",linespacing=1.3)
    save(fig,Path(output),"figure1_workflow")


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    workflow(parser.parse_args().output)
