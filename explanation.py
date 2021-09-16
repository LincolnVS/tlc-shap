
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import shap
import utils
from tqdm import tqdm

import matplotlib
from pathlib import Path
import pickle
import pandas  as pd

def convert_list_string_to_list_float(strings):
    # Converting string to list
    values = []
    for s in strings:
        value = []
        for x in s.split( ):
            try:
                value.append(float(x.strip(' []')))
            except:
                pass
        values.append(np.array(value).reshape(8))
    return values

def load_states(path,num_states=5000):
    buffer = pd.read_csv(f"{path}buffer.csv",sep=";")
    states = convert_list_string_to_list_float(np.random.choice(buffer['actual_state'],num_states))
    return np.array(states)

def load_model_to_be_explained(path):
    return pickle.load(open(f"{path}model.pickle", "rb"))

class shap_explainer_model:

    def __init__ (self, path:str="out/", auto_save:bool=True,force_new:bool=False):
        #Define Vars
        self._model_to_be_explained = None
        self.shap_values = None
        self.expected_value = None
        self._explainer = None
        self._explainer_func = None
        self.explainer_type = None

        self.path = path
        self.auto_save = auto_save
        self.force_new = force_new

        #Create Path to save 
        Path(self.path).mkdir(parents=True, exist_ok=True)    

        #Create or Load shap_explainer_model
        if not force_new:
            try:
                self.update(self.load_model(self.path))
            except:
                pass
        
        self.save_model(self.path)
        #end
    
    def get_shap_values(self,model_to_be_explained,possibilits,new_shap_values:bool=False):
        if self._explainer_func == None:
            raise Exception("Explainer iqual None... Before get shap values, run set_explainer(type)")
        
        self._model_to_be_explained = model_to_be_explained
        
        #print(self.shap_values)
        if self.shap_values == None or new_shap_values:
            self._explainer = self._explainer_func(self._model_to_be_explained,possibilits)
            self.shap_values = self._explainer.shap_values(possibilits)
            self.expected_value = self._explainer.expected_value

        self._explainer = None
        self.save_model(self.path) if self.auto_save else None

        return self.shap_values

    def get_expected_value(self,model_to_be_explained,possibilits,new_expected_value:bool=False):
        if self._explainer_func == None:
            raise Exception("Explainer iqual None... Before get shap values, run set_explainer(type)")
        
        self._model_to_be_explained = model_to_be_explained
        
        #print(self.expected_value)
        if self.expected_value is None or new_expected_value:
            self._explainer = self._explainer_func(self._model_to_be_explained,possibilits)
            self.expected_value = self._explainer.expected_value

        self._explainer = None
        self.save_model(self.path) if self.auto_save else None

        return self.expected_value

    def _get_real_explainer_type(self,type):
        type_lower = type.lower()
        if "deep" in type_lower:
            return "DeepExplainer"
        elif "tree" in type_lower:
            return "TreeExplainer"
        elif "linear" in type_lower:
            return "LinearExplainer"
        elif "kernel" in type_lower:   
            return "KernelExplainer"
        else:
            return None

    def set_explainer(self,type):
        self.explainer_type = self._get_real_explainer_type(type)
        self._explainer_func = getattr(shap, self.explainer_type)

        self.save_model(self.path) if self.auto_save else None
        return self._explainer

    def save_model(self,path=""):
        pickle.dump(self, file = open(f"{path}shap_model.pickle", "wb"))

    def load_model(self,path=""):
        return pickle.load(open(f"{path}shap_model.pickle", "rb"))
    
    def update(self,new_class):
        self._model_to_be_explained = new_class._model_to_be_explained
        self.shap_values = new_class.shap_values
        self.expected_value = new_class.expected_value
        self._explainer = new_class._explainer
        self._explainer_func = new_class._explainer_func
        self.explainer_type = new_class.explainer_type

        self.path = new_class.path 
        self.auto_save = new_class.auto_save
        self.force_new = new_class.force_new


#Gera os estados
path = "tst/"
samples = load_states(path)

#Carrega modelo do agente
model_to_be_explained = load_model_to_be_explained(path)

#Cria agente shap (ou faz load do modelo ja criado)
shap_model = shap_explainer_model(path)

#Define que é deepexplainer
shap_model.set_explainer("deep")
#print(shap_model.explainer_type)

#Gera valores shap (ou pega os que ja estão carregados)
shap_model.get_shap_values(model_to_be_explained,samples,True)

#Gera expected_value
#shap_model.get_expected_value(model_to_be_explained,samples,False)

#Printa valores shap
#print(samples[5],model_to_be_explained.predict(samples[5].reshape(1,8)),shap_model.shap_values[5])

predictions = model_to_be_explained.predict(samples)

#### Começa plots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
sns.set_theme(style="darkgrid")

def hex_to_rgb(hex_color):
    hex_color = hex_color.split('#')[1]
    rgb = list(int(hex_color[i:i+2], 16)/256 for i in (0, 2, 4))
    rgb.append(1)
    return rgb
def create_tab_b(total=8,num = 1):
    cores_hex = ['#2cbdfe','#3aacf6','#489bee','#568ae6','#6379de', '#6e6cd8',
                '#7168d7', '#7f57cf','#8d46c7','#9739c1','#9b35bf','#a924b7', '#b317b1']
    
    #cores_hex = ['#000080','#008000','#A0522D','#4B0082','#FF00FF', '#DC143C',
    #            '#FA8072', '#FFD700']   
    colors = []
    sub_total = 0
    jump_total = 0
    while sub_total < total:
        for sub_num in range(num):
            if sub_total >= total:
                break
            colors.append(hex_to_rgb(cores_hex[jump_total]))
            sub_total += 1 
        jump_total += 1

    return colors

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    #subax.axis('off')
    
    plt.title("Actions")
    
    hexbins = ax.hexbin(list(range(0,8)),[-1]*8, C=colors, 
                     bins=8, gridsize=8,cmap=cm)
    cmin, cmax = hexbins.get_clim()
    below = 0.25 * (cmax - cmin) + cmin
    above = 0.75 * (cmax - cmin) + cmin

    cbar = fig.colorbar(hexbins, cax=subax, orientation='vertical')
    #subax.xaxis.set_ticks_position('top')

    return subax



from colour import Color
import matplotlib as mpl

#red = Color("lightblue")
#colors = list(red.range_to(Color("darkblue"),60))
#colors = [color.rgb for color in colors]
colors = create_tab_b()

cm = LinearSegmentedColormap.from_list(
        'cmap_name', colors, N=8)

for row_index in range(1):

    state = samples[row_index]

    shap.multioutput_decision_plot(
        base_values = list(shap_model.expected_value), 
        shap_values = list(shap_model.shap_values),
        row_index = row_index,
        features=[f'F0: {state[0]}',f'F1: {state[1]}',f'F2: {state[2]}',f'F3: {state[3]}',f'F4: {state[4]}',f'F5: {state[5]}',f'F6: {state[6]}',f'F7: {state[7]}'], #None,
        feature_names=None,
        feature_order=None, #"importance", #
        feature_display_range=None,
        highlight=None, #[np.argmax(predictions[row_index])],#
        link="identity",
        plot_color=plt.get_cmap("tab20c"),#None,
        axis_color="#333333",
        y_demarc_color="#333333",
        alpha=None,
        color_bar=False,#True,
        auto_size_plot=True,
        #title=f'F0:{possibilits[row_index][0]} | F1:{possibilits[row_index][1]} | F2:{possibilits[row_index][2]} | Action={actions[row_index]}',#None,
        xlim=None,
        show=False,#True,
        return_objects=False,
        ignore_warnings=False,
        new_base_value=None,
        legend_labels=None,
        legend_location="best",
    )

    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(20,10)
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(False)
    #plt.title(f'F0:{possibilits[row_index][0]} | F1:{possibilits[row_index][1]} | F2:{possibilits[row_index][2]} | Action={actions[row_index]}')

    for idx,a in enumerate(plt.gca().get_lines()[8:]):
        a.set_color(colors[idx])
        a.set_linewidth(1)
        a.set_alpha(1)

        ##Essa marca alguma linha de fase desejada
        #a.set_alpha(.5)
        #if idx in [9,np.argmax(predictions[row_index])]:
        #    a.set_alpha(1)
        #    a.set_linewidth(3)
        #    a.set_linestyle("-.")


    rect = [0.93,0.05,.05,0.7]
    ax1 = add_subplot_axes(ax,rect)

    plt.savefig(f"{path}multoutput.pdf",bbox_inches = 'tight',pad_inches = 0)
    plt.show()