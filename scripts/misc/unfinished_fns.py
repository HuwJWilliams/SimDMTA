# Unfinished from analysis_class.py
def Loadings_Plot(self,
                    loadings: pd.DataFrame,
                    save_data: bool=False,
                    plot_fname: str='PCA_Loadings',
                    n_components: int=5):
    
    
    if isinstance(loadings, str):
        loadings = pd.read_csv(loadings, index_col='Features')

    loadings['Ref'] = [n for n in range(1, len(loadings)+1)]

    fig = plt.figure(figsize=(30,30))
    gs = gridspec.GridSpec(nrows=n_components, ncols=n_components)

    min_x, max_x = loadings.iloc[:, :n_components].min().min(), loadings.iloc[:, :n_components].max().max()
    for i in range(n_components):
        for j in range(n_components):
            ax = fig.add_subplot(gs[i, j])
            x_pc = f"PC{i+1}"
            y_pc = f"PC{j+1}"

            threshold = 0.4

            filtered_loadings = loadings[
                (loadings['PC1'].abs() > threshold) | (loadings['PC2'].abs() > threshold)
                ]
            
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)

            scatter = sns.scatterplot(
                        x=filtered_loadings[x_pc],
                        y=filtered_loadings[y_pc],
                        s=50,
                        legend=None,
                        ax=ax
                    )
            
            ax.set_xlim(min_x - 0.1 , max_x + 0.1)
            ax.set_ylim(min_x - 0.1, max_x + 0.1)

            if j != 0:
                ax.set_yticklabels([])
            if i != n_components - 1:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(f"{y_pc} Loadings")
            if i == n_components - 1:
                ax.set_xlabel(f"{x_pc} Loadings")

            for ref, (feature, (x,y)) in zip(
                filtered_loadings['Ref'], filtered_loadings[[x_pc, y_pc]].iterrows()
                ):
                scatter.text(
                    x,
                    y, 
                    ref,
                    fontsize=12, 
                    ha='right', 
                    va='top',
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
                    )

            ax.grid(True, linestyle=':', color='gray', alpha=0.5)

            plt.tight_layout()
