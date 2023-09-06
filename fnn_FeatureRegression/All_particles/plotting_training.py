from pylatex import Document, Section, Subsection, Command, Package
from pylatex.utils import NoEscape

# Create a new LaTeX document
doc = Document()
doc.packages.append(Package("listings"))
doc.packages.append(Package("color"))

# Define the listings style
doc.preamble.append(Command('definecolor', 'codegreen', 'rgb', '0,0.6,0'))
doc.preamble.append(Command('definecolor', 'codegray', 'rgb', '0.5,0.5,0.5'))
doc.preamble.append(Command('definecolor', 'codepurple', 'rgb', '0.58,0,0.82'))
doc.preamble.append(Command('definecolor', 'backcolour', 'rgb', '0.95,0.95,0.92'))

doc.preamble.append(NoEscape(r"""
\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\small,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}
\lstset{style=mystyle}
"""))

# Add the code to the document
code = """
out_feats=flat_output_vars

residuals=np.array([])
y_total=np.array([])
y_pred_total=np.array([])
for i, (x,y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    y_pred = model(x)
    y_pred_total=np.append(y_pred_total, y_pred.cpu().detach().numpy())
    y_total=np.append(y_total, y.cpu().detach().numpy())

    # residuals=np.append(residuals, y_pred.cpu().detach().numpy() - y.cpu().detach().numpy())
    # residuals.append(y_pred.cpu().detach().numpy() - y.cpu().detach().numpy())

numfeatures=len(out_feats)
y_pred_total = y_pred_total.reshape(-1,numfeatures)
y_total = y_total.reshape(-1,numfeatures)

print(y_pred_total.shape)
print(y_total.shape)

residuals = [[] for _ in range(numfeatures)]
label_values = [[] for _ in range(numfeatures)]

for i in range(numfeatures):
    y_curr=y_total[:,i]
    # print("ycurr shape before reshape", y_curr.shape)
    # print("ycurr shape after reshape", y_curr.shape)
    y_pred_curr=y_pred_total[:,i]
    # y_pred_curr=y_pred_curr.reshape(-1,1)
    # print("ypredcurr shape after reshape", y_pred_curr.shape)
    residuals_curr = y_pred_curr - y_curr
    residuals[i]=residuals_curr
    label_values[i]=y_curr

residuals = [np.array(res_list) for res_list in residuals]  # Convert lists of arrays to arrays
# residual_medians = [np.median(res) for res in residuals]
residual_std_devs = [np.std(res) for res in residuals]
residual_means = [np.mean(res) for res in residuals]

num_rows = numfeatures // 2 + 1
num_cols = 3

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(150, 150))
flat_axes = axes.flatten()

for i, ax in enumerate(flat_axes[:numfeatures]):
    ax.hist(residuals[i], bins=100, edgecolor='k', alpha=0.65)
    ax.axvline(x=residual_means[i] + residual_std_devs[i], color='r', linestyle='--', label=f'+1 std = {residual_means[i] + residual_std_devs[i]:.2f})')
    ax.axvline(x=residual_means[i] - residual_std_devs[i], color='b', linestyle='--', label=f'-1 std = {residual_means[i] - residual_std_devs[i]:.2f})')
    ax.set_title(f'Residuals for {out_feats[i]}')
    ax.set_yscale('log')
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Display the mean value on the plot
    mean_text = f"Mean: {residual_means[i]:.2f}, std: {residual_std_devs[i]:.5f}"
    ax.text(0.6, 0.85, mean_text, transform=ax.transAxes)

for ax in flat_axes[numfeatures:]:
    ax.axis('off')

# plt.tight_layout()
plt.savefig(f'{prefix}_residuals.png')
plt.show()
"""

with doc.create(Section('Python Code')):
    doc.append(NoEscape(r"\begin{lstlisting}[language=Python]"))
    doc.append(NoEscape(code))
    doc.append(NoEscape(r"\end{lstlisting}"))

# Save the LaTeX file to a temporary location
latex_file_path = "/mnt/data/code_latex.tex"
doc.generate_tex(latex_file_path)

latex_file_path
