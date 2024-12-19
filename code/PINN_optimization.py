import scipy.io as scio
from neural_network import  *
from train_gpt import *
import pickle
from calculate_terms import *
import warnings
from torch.utils.data import TensorDataset,DataLoader

'''
After discovering the structure of the PDE, we use PINN to further optimize the coefficients
'''

warnings.filterwarnings('ignore')
device='cuda'

class PINNLossFunc(nn.Module):
    def __init__(self,h_data_choose):
        super(PINNLossFunc,self).__init__()
        self.h_data=h_data_choose
        return


    def forward(self,A,x):
        x=torch.from_numpy(x.astype(np.float32)).to(device)
        RHS =torch.matmul(A[:, 1:],x)
        LHS = A[:, 0]
        MSE = torch.mean((RHS - LHS) ** 2)
        return MSE

def get_coefficients(ulti_sentence,Net,database,variables,epoch):
    terms=ulti_sentence[::2]
    operators=ulti_sentence[1::2]

    A=[]
    A_column=1
    divide_flag=0
    for i in range(len(terms)):
        term=terms[i]
        operator=operators[i]
        word = id2word[term]
        value = calculate_terms(word, Net, database,variables).reshape(-1, )

        if divide_flag==0:
            A_column *= value
        else:
            A_column /= value
            divide_flag=0
        if operator==2:
            A.append(A_column)
            A_column=1
        elif operator==3:
            continue
        elif operator==4:
            divide_flag=1
        elif operator==1:
            A.append(A_column)



    A = np.vstack(A).T
    b = A[:, 0].copy()

    # ========delete inf===============
    # if epoch < 1000:
    #     lr = Lasso(alpha=1e-4)
    #     lr.fit(A[:, 1:], -b)
    #     x= lr.coef_
    # else:
    #x = np.linalg.lstsq(A[:, 1:], -b)[0]
    u, d, v = np.linalg.svd(A, full_matrices=False)
    x = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
    x=-x[1:]
    return x
def get_PDE_terms(ulti_sentence,Net,database,variables):
    terms=ulti_sentence[::2]
    operators=ulti_sentence[1::2]

    A=[]
    A_column=1
    divide_flag=0
    for i in range(len(terms)):
        term=terms[i]
        operator=operators[i]
        word = id2word[term]
        value = calculate_terms_PINN(word, Net, database,variables).reshape(-1, )
        if divide_flag==0:
            A_column *= value
        else:
            A_column /= value
            divide_flag=0
        if operator==2:
            A.append(A_column.reshape(-1,1))
            A_column=1
        elif operator==3:
            continue
        elif operator==4:
            divide_flag=1
        elif operator==1:
            A.append(A_column.reshape(-1,1))



    A = torch.concatenate(A,axis=1)
    return A
def get_mask_invalid(variables):
    mask_invalid = torch.ones(len(id2word)).to(device)
    if 't' not in variables:
        for i in range(len(id2word)):
            if 't' in id2word[i]:
                mask_invalid[i] = 0
    if 'x' not in variables:
        for i in range(len(id2word)):
            if 'x' in id2word[i]:
                mask_invalid[i] = 0

    if 'y' not in variables:
        for i in range(len(id2word)):
            if 'y' in id2word[i]:
                mask_invalid[i]=0
            if 'Laplace' in id2word[i]:
                mask_invalid[i]=0
            if 'BiLaplace' in id2word[i]:
                mask_invalid[i]=0
            if 'Div' in id2word[i]:
                mask_invalid[i]=0
    if 'z' not in variables:
        for i in range(len(id2word)):
            if 'z' in id2word[i]:
                mask_invalid[i] = 0
            if 'Div' in id2word[i]:
                mask_invalid[i]=0
    return mask_invalid
def get_no_boundary(data_path,Equation_name='',delete_num=3):
    if Equation_name == 'Laplacian_shuttle':
        data = pd.read_csv(data_path, sep='\t')
        data['values'][data['values'] == 'Indeterminate'] = np.NaN
        un = np.array(data.values[:, 3]).astype('float32')

        un = un.reshape([301, 201, 101])
        x = np.arange(-7.65, 7.04 + 14.69 / 300, 14.69 / 300)
        y = np.arange(-4.68, 4.68 + 9.36 / 200, 9.36 / 200)
        z = np.arange(-1.35, 4.16 + 5.51 / 100, 5.51 / 100)
        reserve_index = []
        x_reserve = []
        y_reserve = []
        z_reserve = []
        u_reserve = []
        for i in range(delete_num, un.shape[0] - delete_num):
            for j in range(delete_num, un.shape[1] - delete_num):
                for k in range(delete_num, un.shape[2] - delete_num):
                    if np.isnan(un[i - delete_num:i + delete_num, j - delete_num:j + delete_num,
                                k - delete_num:k + delete_num]).any() == False:
                        reserve_index.append([i, j, k])
        for index in reserve_index:
            x_reserve.append(x[index[0]])
            y_reserve.append(y[index[1]])
            z_reserve.append(z[index[2]])
            u_reserve.append(un[index[0], index[1], index[2]])
        dataset_reserve = np.vstack((x_reserve, y_reserve, z_reserve)).T
        un_reserve = np.array(u_reserve).reshape(-1, 1)
    if Equation_name in ['Burgers_2D','Laplacian_H']:
        data = pd.read_csv(data_path, sep='\t')
        data['values'][data['values'] == 'Indeterminate'] = np.NaN
        un = np.array(data.values[:,3]).astype('float32')

        un = un.reshape([201, 201,101])
        t = np.arange(0, 2+2/200, 2/200)
        x = np.arange(0, 2+2/200, 2/200)
        y=np.arange(0, 1+1/100, 1/100)
        reserve_index=[]
        x_reserve=[]
        y_reserve=[]
        t_reserve=[]
        u_reserve=[]
        for i in range(delete_num,un.shape[0]-delete_num):
            for j in range(delete_num,un.shape[1]-delete_num):
                for k in range(delete_num,un.shape[2]-delete_num):
                    if np.isnan(un[i-delete_num:i+delete_num,j-delete_num:j+delete_num,k-delete_num:k+delete_num]).any()==False:
                        reserve_index.append([i,j,k])
        for index in reserve_index:
            t_reserve.append(t[index[0]])
            x_reserve.append(x[index[1]])
            y_reserve.append(y[index[2]])
            u_reserve.append(un[index[0],index[1],index[2]])
        dataset_reserve=np.vstack((t_reserve,x_reserve, y_reserve)).T
        un_reserve=np.array(u_reserve).reshape(-1,1)
    return dataset_reserve,un_reserve,reserve_index
#============Params=============
Equation_name='KdV_equation'
choose=10000
noise_level=50
noise_type='Gaussian' #Gaussian or Uniform
trail_num='PIS'
Learning_Rate=0.001
canonical_PDEs=['Wave_equation', 'Burgers_equation','KdV_equation','Chaffee_Infante_equation',
                     'KG_equation','KS_equation','Allen_Cahn','Convection_diffusion_equation',
                     'Convection_diffusion_equation','PDE_divide','Eq_6_2_12','PDE_compound']
irrgular_2D_regions=['Possion_x_y','Laplacian_smile','Laplacian_EITech']
irrgular_3D_regions=['Laplacian_shuttle']
temporal_3D_PDEs=['Laplacian_H','Burgers_2D']
#============Get origin data===========
if Equation_name=='Wave_equation':
    # 读取数据
    data_path = f'data/{Equation_name}/wave.mat'
    data = scio.loadmat(data_path)
    un = data.get("u")
    x = np.squeeze(data.get("x"))
    t = np.squeeze(data.get("t").reshape(1, 321))
    x_low = 0.1
    x_up = 3
    t_low = 0.2
    t_up = 6
    target=[[2]]
    Left = 'u_tt'
    epi = 1e-2
    Delete_equation_name='(1.2.10)'
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='Burgers_equation':
    data_path= f'data/{Equation_name}burgers_sine.mat'
    data=scio.loadmat(data_path)
    un=data.get("usol")
    x=np.squeeze(data.get("x"))
    t=np.squeeze(data.get("t").reshape(1,201))
    x_low = -10
    x_up = 10
    t_low = 0
    t_up = 10
    target=[[2],[0,1]]
    Left = 'u_t'
    epi = 5e-4
    Delete_equation_name = 'Bateman-Burgers equation'
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='KdV_equation':
    data_path = f'data/{Equation_name}/KdV-PINN.mat'
    data = scio.loadmat(data_path)
    un = data.get("uu")
    x = np.squeeze(data.get("x"))
    t = np.squeeze(data.get("tt").reshape(1, 201))
    x_low = -0.8
    x_up = 0.8
    t_low = 0.1
    t_up = 0.9
    target = [[3], [0, 1]]
    Left = 'u_t'
    epi = 1e-1
    Delete_equation_name='KdV equation'
    Activation_function = "Sin"  # 'Tanh','Rational'
if Equation_name=='Chaffee_Infante_equation':
    un = np.load(f"data/{Equation_name}/CI.npy")
    x =  np.load(f"data/{Equation_name}/x.npy")
    t =  np.load(f"data/{Equation_name}/t.npy")
    x_low = 0.5
    x_up = 2.5
    t_low = 0.15
    t_up = 0.45
    target = [[2], [0],[0,0,0]]
    Left = 'u_t'
    Delete_equation_name='Chafee–Infante equation'
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='KG_equation':
    data_path = f'data/{Equation_name}/KG_Exp.mat'
    data = scio.loadmat(data_path)
    un = data.get("usol")
    x = np.squeeze(data.get("x"))
    t = np.squeeze(data.get("t").reshape(1, 201))
    x_low = -0.8
    x_up = 0.8
    t_low = 0.3
    t_up = 2.7
    target = [[2], [0]]
    Left = 'u_tt'
    epi=0.1
    #epi=0.001
    Delete_equation_name='Klein–Gordon_u'
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='Allen_Cahn':
    data_path = f'data/{Equation_name}/Allen_Cahn.mat'
    data = scio.loadmat(data_path)
    un = data.get("usol")
    x = np.squeeze(data.get("x"))
    t = np.squeeze(data.get("t").reshape(1, 201))
    x_low = -0.8
    x_up = 0.8
    t_low = 1
    t_up = 9
    target = [[2], [0],[0,0,0]]
    Left = 'u_t'
    epi = 5e-4
    Delete_equation_name='Chafee–Infante equation'
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='Convection_diffusion_equation':
    target = [[2], [1]]
    Left = 'u_t'
    epi=1e-2

    data_path = f'data/{Equation_name}/data.mat'
    data = scio.loadmat(data_path)
    un = data.get("u").T
    x = np.squeeze(data.get("x"))
    t = np.squeeze(data.get("t"))
    x_low=0
    x_up=2
    t_low=0
    t_up=1
    Delete_equation_name='6.2.13'
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='PDE_divide':
    #Also Eq.(8.14.1d)
    target = 'ut+ux/x+0.25uxx=0'
    Left = 'u_t'
    epi=1e-2

    data_path = f'data/{Equation_name}/PDE_divide.npy'
    un= np.load(data_path).T
    x=np.linspace(1,2,100)
    t=np.linspace(0,1,251)
    x_low=1.1
    x_up=1.9
    t_low=0.1
    t_up=0.9
    Delete_equation_name='8.14.1d'
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='Eq_6_2_12':
    data_path= f'data/{Equation_name}/data_Eq_6_2_12.csv'
    un=pd.read_csv(data_path,header=None).values
    un=un.T
    x=np.arange(0,5.01,0.01)
    t=np.arange(0,10.02,0.02)
    x_low = 0.2
    x_up = 4.8
    t_low = 0.4
    t_up = 9.6
    target='0.1*uxt+0.1*ux+ut=0'
    Left = 'u_t'
    epi = 1e-2
    Delete_equation_name = '6.2.12'
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='PDE_compound':
    data_path= f'data/{Equation_name}/PDE_compound.csv'
    un = pd.read_csv(data_path, header=None).values
    un = un.T
    x = np.arange(0, 1.005, 0.005)
    t = np.arange(0, 1.005, 0.005)
    #print(un.shape,x.shape,t.shape)
    x_low = 0.1
    x_up = 0.9
    t_low = 0.05
    t_up = 0.45
    target='ut+0.2(uux)x=0'
    Left = 'u_t'
    epi = 1e-3
    Delete_equation_name = ''
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='Possion_x_y':
    data_path = f'data/Possion_equation/{Equation_name}.xlsx'
    data = pd.read_excel(data_path).values
    x=data[:,0]
    y=data[:,1]
    r_equ=data[:,2]
    theta_equ=data[:,3]
    r=np.arange(0.0001, 1.5, 1.5 / 200)
    theta=np.arange(0, 2*math.pi+2*math.pi/200, 2*math.pi/200)
    un=data[:,4]
    r_low=0.5
    r_up=1.45
    theta_low=0
    theta_up=2*math.pi
    Delete_equation_name = '1.5.20b'
    Activation_function = "Sin"  # 'Tanh','Rational'

if Equation_name=='Laplacian_smile':
    data_path = f'data/{Equation_name}/Laplacian_smile.xlsx'
    data = pd.read_excel(data_path)
    data_plot = pd.read_excel(data_path)
    data=data[data['values']!='Indeterminate']
    data=data.values
    x = np.array(data[:, 0].astype('float32'))
    y = np.array(data[:, 1].astype('float32'))
    un= np.array(data[:,2]).astype('float32')
    data_plot['values'][data_plot['values'] == 'Indeterminate'] = np.NaN
    plot_un= np.array(data_plot.values[:, 2]).astype('float32')
    x_low=-3
    x_up=-2.9
    y_low=-0.1
    y_up=0
    Delete_equation_name = '1.5.20b'
    Activation_function = 'Rational'  # 'Tanh','Rational'

if Equation_name=='Laplacian_EITech':
    data_path = f'data/{Equation_name}/Laplacian_EITech.xlsx'
    data = pd.read_excel(data_path)
    data_plot = pd.read_excel(data_path)
    data=data[data['values']!='Indeterminate']
    data=data.values
    x = np.array(data[:, 0].astype('float32'))
    y = np.array(data[:, 1].astype('float32'))
    un= np.array(data[:,2]).astype('float32')
    print(x.shape,y.shape,un.shape)
    Delete_equation_name = '1.5.20b'
    Activation_function = "Sin"  # 'Tanh','Rational'
if Equation_name=='Laplacian_shuttle':
    data_path = f'data/{Equation_name}/Laplacian_shuttle.dat'
    data = pd.read_csv(data_path,sep='\t')
    data_plot = pd.read_csv(data_path)
    data=data[data['values']!='Indeterminate']
    data=data.values
    x = np.array(data[:, 0].astype('float32'))
    y = np.array(data[:, 1].astype('float32'))
    z= np.array(data[:, 2].astype('float32'))
    un= np.array(data[:,3]).astype('float32')
    Delete_equation_name = '1.8.1'
    Activation_function = 'Rational'  # 'Tanh','Rational'
if Equation_name=='Laplacian_H':
    data_path= f'{Equation_name}/Laplacian_H.dat'
    data = pd.read_csv(data_path, sep='\t')
    data_plot = pd.read_csv(data_path)
    data = data[data['values'] != 'Indeterminate']
    data = data.values

    t = np.array(data[:, 0].astype('float32'))
    x = np.array(data[:, 1].astype('float32'))
    y = np.array(data[:, 2].astype('float32'))
    un = np.array(data[:, 3]).astype('float32')*10
    t_low=0.2
    t_up=1.8
    y_low=0.1
    y_up=0.9
    x_low=0.2
    x_up=1.8
    Delete_equation_name='(1.4.3)'
    Activation_function = "Rational"  # 'Tanh','Rational'
if Equation_name=='Burgers_2D':
    data_path= f'{Equation_name}/Burgers2D.mat'
    data=scio.loadmat(data_path)
    x = np.squeeze(data["x"])
    y = np.squeeze(data["y"])
    t = np.squeeze(data["t"])
    un = data["u"]
    x_low = -0.8
    x_up = 0.8
    t_low = 0
    t_up = 2
    y_low=-0.8
    y_up=0.8
    target=[[2],[0,1]]
    Left = 'u_t'
    epi = 1e-6
    Delete_equation_name='Burgers_2D'
    Activation_function = "Rational"  # 'Tanh','Rational'


#======settings===========
if Equation_name in canonical_PDEs:
    x_num=x.shape[0]
    t_num=t.shape[0]
    total=x_num*t_num
    choose_validate=5000
    meta_data_num=10000

    if noise_type=='Uniform':
        for j in range(x_num):
            for i in range(t_num):
                un[j, i] = un[j, i] * (1 + 0.01 * noise_level * np.random.uniform(-1, 1))
    if noise_type == 'Gaussian':
        noise_value = (noise_level / 100) * np.std(un) * np.random.randn(*un.shape)
        un = un + noise_value
if Equation_name in irrgular_2D_regions:
    x_num = x.shape[0]
    y_num = y.shape[0]
    total = x_num * y_num

    choose_validate = 1000
    meta_data_num = 10000
    delete_num = 10
    window_num = 10
    if noise_type == 'Uniform':
        for j in range(x_num):
            for i in range(y_num):
                un[j, i] = un[j, i] * (1 + 0.01 * noise_level * np.random.uniform(-1, 1))
    if noise_type == 'Gaussian':
        noise_value = (noise_level / 100) * np.std(un) * np.random.randn(*un.shape)
        un = un + noise_value
if Equation_name=='Laplacian_shuttle':
    x_num = x.shape[0]
    y_num = y.shape[0]
    z_num = z.shape[0]
    if noise_type == 'Gaussian':
        noise_value = (noise_level / 100) * np.std(un) * np.random.randn(*un.shape)
        un = un + noise_value
if Equation_name in temporal_3D_PDEs:
    x_num = x.shape[0]
    y_num = y.shape[0]
    t_num = t.shape[0]
    if noise_type == 'Gaussian':
        noise_value = (noise_level / 100) * np.std(un) * np.random.randn(*un.shape)
        un = un + noise_value


#==========NN setting=============
torch.manual_seed(525)
torch.cuda.manual_seed(525)
if Equation_name in ['Laplacian_shuttle','Laplacian_H','Burgers_2D']:
    Net = NN(Num_Hidden_Layers=5,
             Neurons_Per_Layer=50,
             Input_Dim=3,
             Output_Dim=1,
             Data_Type=torch.float32,
             Device='cuda',
             Activation_Function=Activation_function,
             Batch_Norm=False)
else:
    Net=NN(Num_Hidden_Layers=5,
        Neurons_Per_Layer=50,
        Input_Dim=2,
        Output_Dim=1,
        Data_Type=torch.float32,
        Device='cuda',
        Activation_Function=Activation_function,
        Batch_Norm=False)

def train_PINN(Net,un,x,t):
    try:
        os.makedirs(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})')
    except OSError:
        pass

    try:
        os.makedirs(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})')
        np.save(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})/un_{noise_level}', un)
    except OSError:
        un = np.load(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})/un_{noise_level}.npy')
        print('===load noisy data===')
        pass
    #=========produce random dataset==========
    if Equation_name in canonical_PDEs:
        h_data_choose,h_data_validate,database_choose,database_validate=random_data(total,choose,choose_validate,x,t,un,x_num,t_num)
        variables = ['x', 't']
    if Equation_name in irrgular_2D_regions:
        h_data_choose, h_data_validate, database_choose, database_validate = random_data_complex_region(choose,
                                                                                                        choose_validate,
                                                                                                        x, t, un)
        variables = ['x', 'y']
    database_choose = Variable(database_choose.cuda(),requires_grad=True)
    database_validate = Variable(database_validate.cuda(),requires_grad=True)
    h_data_choose=Variable(h_data_choose.cuda())
    h_data_validate=Variable(h_data_validate.cuda())


    best_epoch=np.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy')[0]
    Net.load_state_dict(
        torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/'+f"Net_{Activation_function}_{best_epoch}.pkl"))
    NN_optimizer = torch.optim.Adam([
        {'params': Net.parameters()},
    ])

    best_sentence_save = pickle.load(
        open(f'result_save/{Equation_name}/{choose}_{noise_level}_{noise_type}/sentences.pkl', 'rb'))
    best_sentence_save=best_sentence_save[4][0:3]
    mask_invalid = get_mask_invalid(variables)
    for best_sentence in best_sentence_save:
        vis_sentence = [id2word[int(id)] for id in best_sentence]
        print("".join(vis_sentence[1:-1]))

        best_sentence.pop(0)
        MSELoss = torch.nn.MSELoss()
        PINNLoss= PINNLossFunc(h_data_choose)
        print(f'===============train Net=================')
        for iter in tqdm(range(500)):
            NN_optimizer.zero_grad()
            prediction = Net(database_choose)
            prediction_validate = Net(database_validate).cpu().data.numpy()
            x=get_coefficients(best_sentence,Net,database_choose,variables,iter)
            A=get_PDE_terms(best_sentence,Net,database_choose,variables)
            loss_data = MSELoss(h_data_choose, prediction)
            loss_PDE=PINNLoss(A,x)
            loss=loss_data+0.01*loss_PDE
            loss_validate = np.sum((h_data_validate.cpu().data.numpy() - prediction_validate) ** 2) / choose_validate
            loss.backward()
            NN_optimizer.step()
            if iter==0:
                print(x)
            if (iter+1)%100==0:
                print(x)

def train_PINN_3D(Net,un,x,y,t):
    try:
        os.makedirs(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})')
    except OSError:
        pass

    try:
        os.makedirs(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})')
        np.save(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})/un_{noise_level}', un)
    except OSError:
        un = np.load(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})/un_{noise_level}.npy')
        print('===load noisy data===')
        pass
    #=========produce random dataset==========
    if Equation_name=='Burgers_2D':
        h_data_choose, h_data_validate, database_choose, database_validate = random_data_2D(choose, choose_validate, x, y,
                                                                                            t, un)
    else:
        h_data_choose, h_data_validate, database_choose, database_validate = random_data_H(choose, choose_validate, t,
                                                                                           x, y, un)

    database_choose = Variable(database_choose.cuda(), requires_grad=True)
    database_validate = Variable(database_validate.cuda(), requires_grad=True)
    h_data_choose = Variable(h_data_choose.cuda())
    h_data_validate = Variable(h_data_validate.cuda())
    choose_index=np.random.choice(np.arange(0,database_choose.shape[0],1),10000)

    database_meta, u_reserve, reserve_index = get_no_boundary(data_path, Equation_name, delete_num=15)
    database_meta = torch.from_numpy(database_meta.astype(np.float32))
    database_meta = Variable(database_meta, requires_grad=True).to(device)
    tensor_dataset=TensorDataset(database_choose,h_data_choose)
    data_loader = DataLoader(tensor_dataset, batch_size=50000, shuffle=False)


    best_epoch=np.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy')[0]
    Net.load_state_dict(
        torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/'+f"Net_{Activation_function}_{best_epoch}.pkl"))
    NN_optimizer = torch.optim.Adam([
        {'params': Net.parameters()},
    ])

    best_sentence_save = pickle.load(
        open(f'result_save/{Equation_name}/{choose}_{noise_level}_{noise_type}/sentences.pkl', 'rb'))
    best_sentence_save=best_sentence_save[4][0:3]


    variables = ['x', 'y','t']
    mask_invalid = get_mask_invalid(variables)
    for best_sentence in best_sentence_save:
        vis_sentence = [id2word[int(id)] for id in best_sentence]
        print("".join(vis_sentence[1:-1]))

        best_sentence.pop(0)
        MSELoss = torch.nn.MSELoss()
        PINNLoss= PINNLossFunc(h_data_choose)
        validate_error=[]
        best_validate_error = []
        loss_back = 1e8
        flag = 0
        print(f'===============train Net=================')
        for iter in tqdm(range(100)):
            for step,(batch_x,batch_y) in enumerate(data_loader):
                NN_optimizer.zero_grad()
                prediction = Net(batch_x)
                prediction_validate = Net(database_validate).cpu().data.numpy()
                if Equation_name=='Burgers_2D':
                    x=get_coefficients(best_sentence,Net,database_choose[choose_index.tolist()],variables,iter)
                    A=get_PDE_terms(best_sentence,Net,database_choose[choose_index.tolist()],variables)
                if Equation_name=='Laplacian_H':
                    x = get_coefficients(best_sentence, Net, database_meta, variables, iter)
                    A = get_PDE_terms(best_sentence, Net,  database_meta, variables)
                loss_data = MSELoss(batch_y, prediction)
                loss_PDE=PINNLoss(A,x)
                loss=loss_data+0.01*loss_PDE
                loss_validate = np.sum((h_data_validate.cpu().data.numpy() - prediction_validate) ** 2) / choose_validate
                loss.backward()
                NN_optimizer.step()
            if iter==0:
                print(x)
            if (iter+1)%10==0:
                print(x)


def train_PINN_space_shuttle(Net,un,x,y,z):
    try:
        os.makedirs(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})')
    except OSError:
        pass

    try:
        os.makedirs(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})')
        np.save(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})/un_{noise_level}', un)
    except OSError:
        un = np.load(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})/un_{noise_level}.npy')
        print('===load noisy data===')
        pass
    #=========produce random dataset==========
    h_data_choose, h_data_validate, database_choose, database_validate = random_data_shuttle(choose, choose_validate, x,
                                                                                             y, z, un)

    database_choose = Variable(database_choose.cuda(), requires_grad=True)
    database_validate = Variable(database_validate.cuda(), requires_grad=True)
    h_data_choose = Variable(h_data_choose.cuda())
    h_data_validate = Variable(h_data_validate.cuda())

    tensor_dataset=TensorDataset(database_choose,h_data_choose)
    data_loader = DataLoader(tensor_dataset, batch_size=50000, shuffle=False)
    database_meta, u_reserve, reserve_index = get_no_boundary(data_path, Equation_name, delete_num=15)
    database_meta = torch.from_numpy(database_meta.astype(np.float32))
    database_meta = Variable(database_meta, requires_grad=True).to(device)

    best_epoch=np.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy')[0]
    #best_epoch=80000
    Net.load_state_dict(
        torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/'+f"Net_{Activation_function}_{best_epoch}.pkl"))
    NN_optimizer = torch.optim.Adam([
        {'params': Net.parameters()},
    ])

    best_sentence_save = pickle.load(
        open(f'result_save/{Equation_name}/{choose}_{noise_level}_{noise_type}/sentences.pkl', 'rb'))
    best_sentence_save=best_sentence_save[4][0:3]

    variables = ['x', 'y','z']
    mask_invalid = get_mask_invalid(variables)
    for best_sentence in best_sentence_save:
        vis_sentence = [id2word[int(id)] for id in best_sentence]
        print("".join(vis_sentence[1:-1]))

        best_sentence.pop(0)
        MSELoss = torch.nn.MSELoss()
        PINNLoss= PINNLossFunc(h_data_choose)
        print(f'===============train Net=================')
        for iter in tqdm(range(100)):
            for step,(batch_x,batch_y) in enumerate(data_loader):
                NN_optimizer.zero_grad()
                prediction = Net(batch_x)
                prediction_validate = Net(database_validate).cpu().data.numpy()
                x=get_coefficients(best_sentence,Net,database_meta,variables,iter)
                print(x)
                A=get_PDE_terms(best_sentence,Net,database_meta,variables)
                loss_data = MSELoss(batch_y, prediction)
                loss_PDE=PINNLoss(A,x)
                loss=loss_data+0.01*loss_PDE
                loss_validate = np.sum((h_data_validate.cpu().data.numpy() - prediction_validate) ** 2) / choose_validate
                print(loss_data,loss_PDE)
                loss.backward()
                NN_optimizer.step()
            if iter==0:
                print(x)
            if (iter+1)%10==0:
                print(x)




if __name__=='__main__':
    if Equation_name in canonical_PDEs:
        train_PINN(Net,un,x,t)
    if Equation_name in irrgular_2D_regions:
        train_PINN(Net, un, x,y)
    if Equation_name in temporal_3D_PDEs:
        train_PINN_3D(Net,un,x,y,t)
    if Equation_name=='Laplacian_shuttle':
        train_PINN_space_shuttle(Net,un,x,y,z)

