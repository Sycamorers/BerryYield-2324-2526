"""Report figures computed only from the saved, paired experiment results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_diagrams import style
HERE=Path(__file__).parent
OUT=HERE/'figures'
SCENARIOS=['2324_to_2526','2526_to_2324','within_2324','within_2526']
LABELS={'2324_to_2526':'Train 2324 → evaluate 2526','2526_to_2324':'Train 2526 → evaluate 2324','within_2324':'Within 2324: forward validation','within_2526':'Within 2526: forward validation'}
MODEL_ORDER=['Zero','Train mean','Train median','OLS','Quadratic OLS','Spline ridge','Random forest','SVR','Gradient boosting','Lag OLS','Lag ridge','Lag random forest','RNN','LSTM','Transformer']
FEATURE_ORDER=['Canopy','Roundness','Counts','Canopy + roundness','Canopy + counts','All traits']
COLORS={'Zero':'#889497','Train mean':'#889497','Train median':'#889497','OLS':'#376B91','Quadratic OLS':'#376B91','Spline ridge':'#376B91','Random forest':'#A27543','SVR':'#A27543','Gradient boosting':'#A27543','Lag OLS':'#548235','Lag ridge':'#548235','Lag random forest':'#548235','RNN':'#88678A','LSTM':'#88678A','Transformer':'#88678A'}

def save(fig,stem):
    for ext in ('png','pdf'):fig.savefig(OUT/(stem+'.'+ext),dpi=300)
    plt.close(fig)

def horizon_figure(metrics,scenario):
    a=metrics[(metrics.feature_set=='All traits')&(metrics.scenario==scenario)]
    fig,ax=plt.subplots(figsize=(6.5,2.35))
    for model,color in [('Train mean','#80898A'),('OLS','#376B91'),('Random forest','#A27543'),('Lag ridge','#548235'),('LSTM','#88678A')]:
        s=a[a.model==model].sort_values('horizon_weeks')
        if len(s):ax.plot(s.horizon_weeks,s.MAE,'o--' if model=='Train mean' else 'o-',color=color,label=model,linewidth=1.4,markersize=3.3)
    ax.set(xlabel='Forecast horizon (weeks)',ylabel='MAE (g/plot)',xticks=[1,2,3],ylim=(0,None))
    ax.grid(axis='y',color='#E1E8E9',linewidth=.6)
    ax.legend(frameon=False,ncol=5,fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.2))
    fig.subplots_adjust(left=.12,right=.99,bottom=.23,top=.8)
    save(fig,'horizons_'+scenario)

def ablation_figure(metrics):
    fig,axes=plt.subplots(2,2,figsize=(6.5,4.6),sharex=True)
    all_models=['OLS','Random forest','Lag ridge','LSTM']
    colors=['#376B91','#A27543','#548235','#88678A']
    for ax,sc in zip(axes.flat,SCENARIOS):
        a=metrics[(metrics.scenario==sc)&(metrics.horizon_weeks==1)]
        for model,color in zip(all_models,colors):
            vals=[a.loc[(a.model==model)&(a.feature_set==f),'MAE'].iloc[0] if ((a.model==model)&(a.feature_set==f)).any() else np.nan for f in FEATURE_ORDER]
            ax.plot(np.arange(6),vals,'o-',label=model,color=color,linewidth=1.1,markersize=3)
        ax.set_title(LABELS[sc],fontsize=8.5,weight='bold');ax.set_ylim(bottom=0)
        ax.grid(axis='y',color='#E1E8E9',linewidth=.6);ax.set_ylabel('MAE (g/plot)',fontsize=8)
        ax.set_xticks(np.arange(6),['C','Rd','F','C+Rd','C+F','All'],fontsize=8)
        ax.tick_params(axis='x',labelbottom=True)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.075),ncol=4,frameon=False,fontsize=8)
    fig.text(.5,.02,'C = canopy; Rd = roundness; F = FL/G/W/P + plant tracks',ha='center',fontsize=8)
    fig.subplots_adjust(left=.1,right=.99,bottom=.19,top=.94,hspace=.5,wspace=.28)
    save(fig,'feature_comparison')

def stage_figure(stage_metrics,group):
    scenarios=SCENARIOS[:2] if group=='cross' else SCENARIOS[2:]
    shown=['Train mean','OLS','Random forest','Lag ridge','LSTM']
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.8))
    for ax,sc in zip(axes,scenarios):
        a=stage_metrics[(stage_metrics.feature_set=='All traits')&(stage_metrics.horizon_weeks==1)&(stage_metrics.scenario==sc)]
        for model in shown:
            vals=[a.loc[(a.model==model)&(a.stage==stage),'MAE'].iloc[0] if ((a.model==model)&(a.stage==stage)).any() else np.nan for stage in ['Early','Middle','Late']]
            ax.plot([0,1,2],vals,'o--' if model=='Train mean' else 'o-',label=model,color=COLORS[model],linewidth=1.2,markersize=3)
        ax.set_xticks([0,1,2],['Early','Middle','Late']);ax.set_title(LABELS[sc],fontsize=8.5,weight='bold');ax.set_ylabel('MAE (g/plot)');ax.set_ylim(bottom=0)
        ax.grid(axis='y',color='#E1E8E9',linewidth=.6)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,fontsize=7.3)
    fig.subplots_adjust(left=.1,right=.99,bottom=.25,top=.86,wspace=.32)
    save(fig,'stage_'+group)

def main():
    OUT.mkdir(exist_ok=True);style()
    m=pd.read_csv(HERE/'analysis/metrics.csv')
    for s in SCENARIOS:horizon_figure(m,s)
    ablation_figure(m)
    stages=pd.read_csv(HERE/'analysis/stage_metrics.csv')
    for g in ['cross','within']:stage_figure(stages,g)

if __name__=='__main__':main()


def prediction_diagnostics(metrics,predictions):
    """Show errors against real measured harvests; winners are descriptive only."""
    fig,axes=plt.subplots(2,2,figsize=(6.5,4.5))
    for ax,sc in zip(axes.flat,SCENARIOS):
        a=metrics[(metrics.scenario==sc)&(metrics.feature_set=='All traits')&(metrics.horizon_weeks==1)&(~metrics.model.isin(MODEL_ORDER[:3]))]
        best=a.loc[a.MAE.idxmin()]
        p=predictions[(predictions.scenario==sc)&(predictions.feature_set=='All traits')&(predictions.horizon_weeks==1)&(predictions.model==best.model)].copy()
        p['yield_date']=pd.to_datetime(p.yield_date)
        d=p.groupby('yield_date').agg(observed=('yield_g','mean'),predicted=('prediction','mean'),n=('sample_id','size')).reset_index()
        x=np.arange(len(d))
        ax.plot(x,d.observed,'o-',color='#263A42',label='Observed yield',linewidth=1.5,markersize=3)
        ax.plot(x,d.predicted,'s--',color='#A87445',label='Predicted yield',linewidth=1.3,markersize=3)
        ax.set_title(LABELS[sc]+'\n'+str(best.model)+f'; R²={best.R2:.2f}',fontsize=8.1,weight='bold')
        ax.set_xticks(x,d.yield_date.dt.strftime('%m/%d'),rotation=35,ha='right',fontsize=6.7)
        ax.set_ylabel('Mean yield (g/plot)',fontsize=8);ax.set_ylim(bottom=0)
        ax.grid(axis='y',color='#E1E8E9',linewidth=.6)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.1,right=.99,bottom=.15,top=.92,hspace=.65,wspace=.28)
    save(fig,'observed_predicted')
