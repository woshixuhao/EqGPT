import scipy.io as scio
from neural_network import  *
from train_gpt import *
import matplotlib.pyplot as plt

'''
This code is utilized for training the surrogate model from sparse and noisy data.
The surroagte model is utilized to generate meta-data and calculate derivates.
'''

device='cuda'


def Generate_meta_data(Net,Equation_name, choose, noise_level, trail_num, Load_state, x_low, x_up, t_low, t_up, nx=100,
                       nt=100, ):
    Net.load_state_dict(torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl'))
    Net.eval()

    if Equation_name=='Possion_x_y':
        r = torch.linspace(x_low, x_up, nx)
        theta = torch.linspace(t_low, t_up, nt)
        num = 0
        data = torch.zeros(2)
        database = torch.zeros([theta.shape[0] * r.shape[0], 2])
        print(database.shape)
        for j in range(nx):
            for i in range(nt):
                data[0] = r[j] * torch.cos(theta[i])
                data[1] = r[j] * torch.sin(theta[i])
                database[num] = data
                num += 1
    else:
        x = torch.linspace(x_low, x_up, nx)
        t = torch.linspace(t_low, t_up, nt)
        total = nx * nt

        num = 0
        data = torch.zeros(2)
        h_data = torch.zeros([total, 1])
        database = torch.zeros([total, 2])
        for j in range(nx):
            for i in range(nt):
                data[0] = x[j]
                data[1] = t[i]
                database[num] = data
                num += 1

    database = Variable(database, requires_grad=True).to(device)

    return Net, database
def Generate_meta_data_H(Net,Equation_name, choose, noise_level, trail_num, Load_state, t_low, t_up, y_low,y_up,nx=20,
                       ny=20,nt=20):
    Net.load_state_dict(
        torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl'))
    Net.eval()
    x =torch.concatenate((torch.linspace(0.2,0.8, 10),torch.linspace(1.2,1.8, 10)))
    t = torch.linspace(t_low, t_up, nt)
    y = torch.linspace(y_low, y_up, ny)
    total = nx * nt * ny

    num = 0
    data = torch.zeros(3)
    database = torch.zeros([total, 3])
    for j in range(nt):
        for k in range(nx):
            for i in range(ny):
                data[0] = t[j]
                data[1] = x[k]
                data[2] = y[i]
                database[num] = data
                num += 1

    database = Variable(database, requires_grad=True).to(device)
    return Net, database
def Generate_meta_data_2D(Net,Equation_name, choose, noise_level, trail_num, Load_state, x_low, x_up, t_low, t_up, y_low,y_up,nx=20,
                       ny=20,nt=20):
    Net.load_state_dict(
        torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl'))
    Net.eval()
    x = torch.linspace(x_low, x_up, nx)
    t = torch.linspace(t_low, t_up, nt)
    y = torch.linspace(y_low, y_up, ny)
    total = nx * nt * ny

    num = 0
    data = torch.zeros(3)
    database = torch.zeros([total, 3])
    for j in range(nx):
        for k in range(ny):
            for i in range(nt):
                data[0] = x[j]
                data[1] = y[k]
                data[2] = t[i]
                database[num] = data
                num += 1

    database = Variable(database, requires_grad=True).to(device)
    return Net, database
def get_no_boundary(data_path,Equation_name='',delete_num=3):
    if Equation_name == 'Laplacian_smile':
        data = pd.read_excel(data_path)
        data['values'][data['values'] == 'Indeterminate'] = np.NaN
        un = np.array(data.values[:, 2]).astype('float32')
        un=un.reshape(251,251)
        x = np.arange(-4, 4 + 8 / 250, 8 / 250)
        y = np.arange(-4, 4 + 8 / 250, 8 / 250)
        reserve_index = []
        x_reserve = []
        y_reserve = []
        u_reserve = []
        for i in range(delete_num, un.shape[0] - delete_num):
            for j in range(delete_num, un.shape[1] - delete_num):
                if np.isnan(un[i - delete_num:i + delete_num, j - delete_num:j + delete_num]).any() == False:
                    reserve_index.append([i, j])
        for index in reserve_index:
            x_reserve.append(x[index[0]])
            y_reserve.append(y[index[1]])
            u_reserve.append(un[index[0], index[1]])
        dataset_reserve = np.vstack((x_reserve, y_reserve)).T
        un_reserve = np.array(u_reserve).reshape(-1, 1)
    if Equation_name == 'Laplacian_EITech':
        data = pd.read_excel(data_path)
        data['values'][data['values'] == 'Indeterminate'] = np.NaN
        un = np.array(data.values[:, 2]).astype('float32')
        un = un.reshape(801, 201)
        x = np.arange(4.3, 141 + 136.7 / 800, 136.7/ 800)
        y = np.arange(19.6, 54.6 + 35 / 200, 35 / 200)
        reserve_index = []
        x_reserve = []
        y_reserve = []
        u_reserve = []
        for i in range(delete_num, un.shape[0] - delete_num):
            for j in range(delete_num, un.shape[1] - delete_num):
                if np.isnan(un[i - delete_num:i + delete_num, j - delete_num:j + delete_num]).any() == False:
                    reserve_index.append([i, j])
        for index in reserve_index:
            x_reserve.append(x[index[0]])
            y_reserve.append(y[index[1]])
            u_reserve.append(un[index[0], index[1]])
        dataset_reserve = np.vstack((x_reserve, y_reserve)).T
        un_reserve = np.array(u_reserve).reshape(-1, 1)
    if Equation_name=='Laplacian_shuttle':
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
    return dataset_reserve,un_reserve,reserve_index
#============Params=============
Equation_name='KdV_equation'
#['Wave_equation', 'Burgers_equation','KdV_equation','Chaffee_Infante_equation',
# 'KG_equation','KS_equation','Allen_Cahn','Convection_diffusion_equation',
# 'Convection_diffusion_equation','PDE_divide','Eq_6_2_12','PDE_compound',
# 'Possion_x_y','Laplacian_smile','Laplacian_EITech']
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
    data_path = f'data/{Equation_name}/wave.mat'
    data = scio.loadmat(data_path)
    un = data.get("u")
    x = np.squeeze(data.get("x"))
    t = np.squeeze(data.get("t").reshape(1, 321))
    x_low = 0.1
    x_up = 3
    t_low = 0.2
    t_up = 6
    Left = 'u_tt'
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
    Left = 'u_t'
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
    Left = 'u_t'
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
    Left = 'u_tt'
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
    Left = 
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
#====================Train GPT without target equation========================
'''
This is very important to keep the generative model unseen underlying equations for proof-of-concept
'''
if os.path.exists(f'gpt_model/PDEGPT_{Equation_name}.pt')==False:
    train_num_data = get_train_dataset(Equation_name=Delete_equation_name)
    batch_size = 128
    epochs = 100
    dataset = MyDataSet(train_num_data)

    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)

    model = GPT().to(device)

    train(model, data_loader,Equation_name=Equation_name)




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

def train_surrogate_model(Net,un):
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
    iter_num=50000
    if Equation_name in canonical_PDEs:
        h_data_choose,h_data_validate,database_choose,database_validate=random_data(total,choose,choose_validate,x,t,un,x_num,t_num)
    if Equation_name in irrgular_2D_regions:
        h_data_choose, h_data_validate, database_choose, database_validate = random_data_complex_region(choose, choose_validate,
                                                                                         x,y, un)
    if Equation_name=='Laplacian_shuttle':
        h_data_choose, h_data_validate, database_choose, database_validate = random_data_shuttle(choose,
                                                                                                 choose_validate, x, y,
                                                                                                 z, un)
        iter_num=100000
    if Equation_name=='Laplacian_H':
        h_data_choose, h_data_validate, database_choose, database_validate = random_data_H(choose, choose_validate, t,
                                                                                           x, y, un)
    if Equation_name=='Burgers_2D':
        h_data_choose, h_data_validate, database_choose, database_validate = random_data_2D(choose, choose_validate, x,
                                                                                            y, t, un)

    database_choose = Variable(database_choose.cuda(),requires_grad=True)
    database_validate = Variable(database_validate.cuda(),requires_grad=True)
    h_data_choose=Variable(h_data_choose.cuda())
    h_data_validate=Variable(h_data_validate.cuda())


    torch.save(Net.state_dict(), f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/'+f"Net_{Activation_function}_origin.pkl")

    NN_optimizer = torch.optim.Adam([
        {'params': Net.parameters()},
    ])


    MSELoss = torch.nn.MSELoss()
    validate_error=[]
    print(f'===============train Net=================')
    file = open(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/loss.txt', 'w').close()
    file=open(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/loss.txt',"a+")
    for iter in range(iter_num):
        NN_optimizer.zero_grad()
        prediction = Net(database_choose)
        prediction_validate = Net(database_validate).cpu().data.numpy()
        loss = MSELoss(h_data_choose, prediction)
        loss_validate = np.sum((h_data_validate.cpu().data.numpy() - prediction_validate) ** 2) / choose_validate
        loss.backward()
        NN_optimizer.step()



        if (iter+1) % 500 == 0:
            validate_error.append(loss_validate)
            torch.save(Net.state_dict(),
                       f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/' + f"Net_{Activation_function}_{iter + 1}.pkl")

            print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (iter+1, loss, loss_validate))
            file.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \n" % (iter+1, loss, loss_validate))
    file.close()
    best_epoch=(validate_error.index(min(validate_error))+1)*500
    print(best_epoch)
    np.save(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy',np.array([best_epoch]))


def get_meta(Net):
    best_epoch = np.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy')[
        0]
    print("best_epoch:", best_epoch)
    Load_state = 'Net_' + Activation_function + f'_{best_epoch}'
    if Equation_name in canonical_PDEs:
        Net, database = Generate_meta_data(Net, Equation_name, choose, noise_level, trail_num, Load_state, x_low, x_up,
                                       t_low, t_up)
    if Equation_name in ['Laplacian_smile', 'Laplacian_EITech']:
        database, u_reserve, reserve_index = get_no_boundary(data_path, Equation_name, delete_num=8)
        database = torch.from_numpy(database.astype(np.float32))
        database = Variable(database, requires_grad=True).to(device)
        Net.load_state_dict(
            torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl'))
        Net.eval()
    if Equation_name=='Possion_x_y':
        Net, database, _, _ = Generate_meta_data(Net, Equation_name, choose, noise_level, trail_num, Load_state, r_low,
                                                 r_up, theta_low, theta_up)
    if Equation_name=='Laplacian_shuttle':
        database, u_reserve, reserve_index = get_no_boundary(data_path, Equation_name, delete_num=15)
        database = torch.from_numpy(database.astype(np.float32))
        database = Variable(database, requires_grad=True).to(device)
        Net.load_state_dict(
            torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl'))
        Net.eval()
    if Equation_name=='Laplacian_H':
        Net, database = Generate_meta_data_H(Net, Equation_name, choose, noise_level, trail_num, Load_state,
                                           t_low, t_up, y_low, y_up)
    if Equation_name=='Burgers_2D':
        Net, database = Generate_meta_data_2D(Net, Equation_name, choose, noise_level, trail_num, Load_state, x_low, x_up,
                                           t_low, t_up, y_low, y_up)
    return Net,database


if __name__=='__main__':
    train_surrogate_model(Net,un)

