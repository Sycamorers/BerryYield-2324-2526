"""Clear, report-sized study diagrams; all bitmap exports also have vector PDFs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT=Path(__file__).parent/'figures'
BLUE='#315E77';GREEN='#4B7254';GRAY='#667477';INK='#263A42';LIGHT='#F0F5F6'

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'text.color':INK,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})

def box(ax,x,y,w,h,title,body,color=BLUE,fontsize=8):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.035',linewidth=.9,edgecolor=color,facecolor=LIGHT)
    ax.add_patch(p)
    ax.text(x+w/2,y+h*.73,title,ha='center',va='center',weight='bold',fontsize=9,color=color)
    ax.text(x+w/2,y+h*.35,body,ha='center',va='center',fontsize=fontsize,linespacing=1.4)

def arrow(ax,a,b,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,linewidth=1.2,color=color))

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'.png'),dpi=300)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

def overview():
    style();fig,ax=plt.subplots(figsize=(6.5,3.1));ax.set(xlim=(0,10),ylim=(0,4.4));ax.axis('off')
    box(ax,.12,2.6,2.68,1.5,'1. Rebuild saved traits','Recompute plant roundness\nSum areas + count tracks')
    box(ax,3.55,2.6,2.68,1.5,'2. Match future yield','Canopy + roundness + counts\n1-, 2- and 3-week targets')
    box(ax,7,2.6,2.83,1.5,'3. Compare models','Statistical regression\nMachine learning + sequences')
    arrow(ax,(2.85,3.35),(3.5,3.35));arrow(ax,(6.28,3.35),(6.95,3.35))
    box(ax,.12,.3,4.55,1.5,'Within each season','2324 forward validation\n2526 forward validation',GREEN,9)
    box(ax,5.35,.3,4.48,1.5,'Across seasons, both directions','Train 2324 → evaluate 2526\nTrain 2526 → evaluate 2324',GREEN,9)
    arrow(ax,(8.4,2.55),(8.4,2.1));arrow(ax,(8.4,2.1),(2.4,2.1));arrow(ax,(2.4,2.1),(2.4,1.85));arrow(ax,(7.6,2.1),(7.6,1.85))
    ax.text(4.95,.02,'Report each result for the full season and early / middle / late harvest phases.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.005,right=.995,top=.99,bottom=.035);save(fig,'study_overview')

def evaluation():
    style();fig,axes=plt.subplots(2,1,figsize=(6.5,3.05),gridspec_kw={'height_ratios':[1.15,1]})
    for ax in axes:ax.set(xlim=(0,10),ylim=(0,2));ax.axis('off')
    ax=axes[0];ax.text(0,1.87,'Within-season: only completed earlier harvests may train the model',weight='bold',fontsize=9)
    box(ax,.15,.55,3.4,.8,'Training labels available','Harvest dates before the image date',GREEN,7.5)
    box(ax,4.2,.55,2.2,.8,'Forecast origin','Image features known now',BLUE,7.5)
    box(ax,7.25,.55,2.45,.8,'Future harvest','Observed yield for scoring',BLUE,7.5)
    arrow(ax,(3.62,.95),(4.14,.95));arrow(ax,(6.48,.95),(7.18,.95))
    ax.text(8.48,.2,'Approximately +7, +14 or +21 days',ha='center',fontsize=8)
    ax=axes[1];ax.text(0,1.85,'Across-season: keep the evaluation season out of fitting',weight='bold',fontsize=9)
    box(ax,.15,.25,4.2,1.15,'Train using 2324 only','Evaluate using observed 2526 harvests',GREEN,8)
    box(ax,5.6,.25,4.2,1.15,'Train using 2526 only','Evaluate using observed 2324 harvests',GREEN,8)
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.99,hspace=.15);save(fig,'evaluation_design')

if __name__=='__main__':overview();evaluation()


def image_to_trait_workflow():
    """General image-processing method, without execution-status annotations."""
    style();fig,ax=plt.subplots(figsize=(6.5,2.05));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    ax.text(.1,2.78,'Image-to-feature workflow',fontsize=12,weight='bold',color=INK)
    stages=[
        ('RGB sequences','Images grouped by\nseason, date and plot'),
        ('YOLO + ByteTrack','Detect plants\nand link tracks'),
        ('Representative\nframe','One middle frame\nper plant track'),
        ('SAM segmentation','Plant masks\nArea + perimeter'),
        ('Plot features','Canopy + roundness\nPlant tracks'),
    ]
    for i,(title,body) in enumerate(stages):
        x=.12+2*i
        box(ax,x,.83,1.72,1.35,title,body,color=GREEN,fontsize=7)
        # Short headings and body text keep the report-size diagram readable.
        texts=ax.texts[-2:]
        texts[0].set_fontsize(7.4)
        if i<4:arrow(ax,(x+1.75,1.5),(x+1.96,1.5),color=GRAY)
    ax.text(5,.3,'Plant circularity = 4πA/P²    |    Canopy = sum of plant-mask pixel areas',ha='center',fontsize=8)
    fig.subplots_adjust(left=.005,right=.995,bottom=.02,top=.99);save(fig,'image_to_trait_workflow')
