import numpy as np
import torch
import json
dict_datas = json.load(open('dict_datas_0725.json', 'r'))
word2id = dict_datas["word2id"]
id2word= dict_datas["id2word"]
#print(id2word)

def calculate_terms(word,Net,database,variables):
    if 'x' in variables:
        x_index=variables.index('x')
    if 't' in variables:
        t_index=variables.index('t')
    if 'y' in variables:
        y_index=variables.index('y')
    if 'z' in variables:
        z_index=variables.index('z')

    u = Net(database)
    H_grad = torch.autograd.grad(outputs=u.sum(), inputs=database, create_graph=True)[0]
    Hx = H_grad[:, x_index].reshape(-1,1)
    if 't' in variables:
        Ht = H_grad[:, t_index].reshape(-1,1)
    if word=='ut':
        return Ht.cpu().data.numpy()
    if word=='u':
        return u.cpu().data.numpy()
    if word=='ux':
        return Hx.cpu().data.numpy()
    if word=='uxx':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return Hxx.cpu().data.numpy()
    if word=='uxxt':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        Hxxt=torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, t_index]
        return Hxxt.cpu().data.numpy()
    if word=='ut^2':
        return (Ht**2).cpu().data.numpy()
    if word=='uxt':
        result=torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, t_index]
        return result.cpu().data.numpy()
    if word=='ux^2':
        return (Hx**2).cpu().data.numpy()
    if word=='utt':
        utt=torch.autograd.grad(outputs=Ht.sum(), inputs=database, create_graph=True)[0][:, t_index]
        return utt.cpu().data.numpy()
    if word=='uxxxx':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        Hxxxx = torch.autograd.grad(outputs=Hxxx.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return Hxxxx.cpu().data.numpy()
    if word=='(u^2)xx':
        H2x= torch.autograd.grad(outputs=(Hx**2).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        H2xx = torch.autograd.grad(outputs=H2x.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return H2xx.cpu().data.numpy()
    if word=='(uux)x':
        HHx_x=torch.autograd.grad(outputs=(u*Hx).sum(), inputs=database, create_graph=True)[0][:, x_index]
        return HHx_x.cpu().data.numpy()
    if word=='uxxtt':
        Hxx= torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        Hxxt=torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, t_index].reshape(-1,1)
        Hxxtt=torch.autograd.grad(outputs=Hxxt.sum(), inputs=database, create_graph=True)[0][:, t_index]
        return Hxxtt.cpu().data.numpy()
    if word=='(u^4)xx':
        H4x=torch.autograd.grad(outputs=(u**4).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        H4xx=torch.autograd.grad(outputs=H4x.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return H4xx.cpu().data.numpy()
    if word=='(u^3)x':
        H3x = torch.autograd.grad(outputs=(u ** 3).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        return H3x.cpu().data.numpy()
    if word=='uxxx':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1,1)
        Hxxx= torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return Hxxx.cpu().data.numpy()
    if word=='u^3':
        return (u**3).cpu().data.numpy()
    if word=='x':
        return database[:,x_index].cpu().data.numpy()
    if word=='u^2':
        return (u**2).cpu().data.numpy()
    if word=='(1/u)xx':
        temp=torch.autograd.grad(outputs=(u**(-1)).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        result=torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:, x_index]
        return result.cpu().data.numpy()
    if word=='(u^-2*ux)x':
        result= torch.autograd.grad(outputs=(u**(-2)*Hx).sum(), inputs=database, create_graph=True)[0][:, x_index]
        return result.cpu().data.numpy()
    if word=='ux^2':
        return (Hx**2).cpu().data.numpy()
    if word=='uxxxxx':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hxxxx = torch.autograd.grad(outputs=Hxxx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        Hxxxxx = torch.autograd.grad(outputs=Hxxxx.sum(), inputs=database, create_graph=True)[0][:, x_index]
        del Hxxxx
        return Hxxxxx.cpu().data.numpy()
    if word=='uyy':
        Hy= H_grad[:, y_index].reshape(-1,1)
        Hyy=torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index]
        return (Hyy).cpu().data.numpy()
    if word=='ut^3':
        return (Ht**3).cpu().data.numpy()
    if word=='sqrt(u)':
        return (torch.sqrt(u)).cpu().data.numpy()
    if word=='sin(u)':
        return torch.sin(u).cpu().data.numpy()
    if word=='sinh(u)':
        return torch.sinh(u).cpu().data.numpy()
    if word=='BiLaplace(u)':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        Hxxxx = torch.autograd.grad(outputs=Hxxx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        Hy = H_grad[:,  y_index].reshape(-1,1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyyy = torch.autograd.grad(outputs=Hyyy.sum(), inputs=database, create_graph=True)[0][:,y_index].reshape(-1, 1)
        # if 'z' in variables:
        #     Hz =H_grad[:, z_index].reshape(-1,1)
        #     Hzz = torch.autograd.grad(outputs=Hz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        #     Hzzz = torch.autograd.grad(outputs=Hzz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        #     Hzzzz = torch.autograd.grad(outputs=Hzzz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        #     Hxxy=torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        #     Hxxyy=torch.autograd.grad(outputs=Hxxy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        #     Hyyz= torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:,  z_index].reshape(-1, 1)
        #     Hyyzz= torch.autograd.grad(outputs=Hyyz.sum(), inputs=database, create_graph=True)[0][:,  z_index].reshape(-1, 1)
        #     Hxxz=torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:,  z_index].reshape(-1, 1)
        #     Hxxzz=torch.autograd.grad(outputs=Hxxz.sum(), inputs=database, create_graph=True)[0][:,  z_index].reshape(-1, 1)
        #     result=Hxxxx+Hyyyy+Hzzzz+2*Hxxyy+2*Hyyzz+2*Hxxzz
        # else:
        Hxxy = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1,
                                                                                                                 1)
        Hxxyy = torch.autograd.grad(outputs=Hxxy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(
            -1, 1)
        result = Hxxxx + Hyyyy + 2 * Hxxyy
        del Hxxxx
        del Hyyyy
        del Hxxyy
        return result.cpu().data.numpy()
    if word=='uyyyy':
        Hy = H_grad[:, y_index].reshape(-1, 1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyyy = torch.autograd.grad(outputs=Hyyy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1,
                                                                                                                   1)
        return Hyyyy.cpu().data.numpy()
    if word=='uxxyy':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hxxy = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hxxyy = torch.autograd.grad(outputs=Hxxy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1,
                                                                                                                   1)
        return Hxxyy.cpu().data.numpy()
    if word=='uzzzz':
        Hz = H_grad[:, z_index].reshape(-1, 1)
        Hzz = torch.autograd.grad(outputs=Hz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        Hzzz = torch.autograd.grad(outputs=Hzz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        Hzzzz = torch.autograd.grad(outputs=Hzzz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        return Hzzzz.cpu().data.numpy()
    if word=='uyyzz':
        Hy = H_grad[:, y_index].reshape(-1, 1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyz = torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        Hyyzz = torch.autograd.grad(outputs=Hyyz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1,
                                                                                                                   1)
        return Hyyzz.cpu().data.numpy()
    if word=='uxxzz':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hxxz = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        Hxxzz = torch.autograd.grad(outputs=Hxxz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1,
                                                                                                                   1)
        return Hxxzz.cpu().data.numpy()
    if word=='y':
        return database[:,y_index].cpu().data.numpy()
    if word=='uy':
        Hy =  H_grad[:, y_index].reshape(-1,1)
        return Hy.cpu().data.numpy()
    if word=='x^2':
        return (database[:,x_index]**2).cpu().data.numpy()
    if word=='y^2':
        return (database[:, y_index]**2).cpu().data.numpy()
    if word=='uy^2':
        Hy =  H_grad[:, y_index].reshape(-1,1)
        return (Hy**2).cpu().data.numpy()
    if word=='uxy':
        result= torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        return result.cpu().data.numpy()
    if word=='uz':
        Hz =H_grad[:, z_index].reshape(-1,1)
        return Hz.cpu().data.numpy()
    if word=='Div(u)':
        Hy = H_grad[:, y_index].reshape(-1, 1)
        if 'z' not in variables:
            result = Hx + Hy
        else:
            Hz =H_grad[:, z_index].reshape(-1,1)
            result=Hx+Hy+Hz
        return result.cpu().data.numpy()
    if word=='Laplace(u)':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hy = H_grad[:, y_index].reshape(-1,1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        if 'z' in variables:
            Hz =H_grad[:, z_index].reshape(-1,1)
            Hzz = torch.autograd.grad(outputs=Hz.sum(), inputs=database, create_graph=True)[0][:,z_index].reshape(-1, 1)
            result = Hxx + Hyy + Hzz
        else:
            result = Hxx + Hyy
        return result.cpu().data.numpy()
    if word=='uzz':
        Hz = H_grad[:, z_index].reshape(-1, 1)
        Hzz = torch.autograd.grad(outputs=Hz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        return Hzz.cpu().data.numpy()
    if word=='Laplace(utt)':
        utt = torch.autograd.grad(outputs=Ht.sum(), inputs=database, create_graph=True)[0][:, t_index]
        uttx= torch.autograd.grad(outputs=utt.sum(), inputs=database, create_graph=True)[0][:, x_index]
        uttxx = torch.autograd.grad(outputs=uttx.sum(), inputs=database, create_graph=True)[0][:,x_index]
        utty = torch.autograd.grad(outputs=utt.sum(), inputs=database, create_graph=True)[0][:, y_index]
        uttyy = torch.autograd.grad(outputs=utty.sum(), inputs=database, create_graph=True)[0][:, y_index]
        if 'z' in variables:
            uttz = torch.autograd.grad(outputs=utt.sum(), inputs=database, create_graph=True)[0][:, z_index]
            uttzz = torch.autograd.grad(outputs=uttz.sum(), inputs=database, create_graph=True)[0][:, z_index]
            result=uttxx+uttyy+uttzz
        else:
            result=uttxx+uttyy
        return result.cpu().data.numpy()
    if word=='ut^3':
        return (Ht**3).cpu().data.numpy()
    if word=='(x+y)':
        return (database[:,x_index]+database[:,y_index]).cpu().data.numpy()
    if word=='exp(x)':
        return torch.exp(database[:,x_index]).cpu().data.numpy()
    if word=='uyyt':
        Hy = H_grad[:, 2].reshape(-1,1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyt=torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        return Hyyt.cpu().data.numpy()
    if word=='sint':
        return torch.sin(database[:,x_index]).cpu().data.numpy()
    if word=='sinx':
        return torch.sin(database[:, x_index]).cpu().data.numpy()
    if word=='(uxx+ux/x)^2':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result=(Hxx+(1/database[:,x_index].reshape(-1,1))*Hx)**2
        return  result.cpu().data.numpy()
    if word=='x^4':
        return (database[:, x_index]**4).cpu().data.numpy()
    if word=='sqrt(x)':
        return (torch.sqrt(database[:, x_index])).cpu().data.numpy()
    if word=='exp(-y)':
        return (torch.exp(-database[:, y_index])).cpu().data.numpy()
    if word=='t':
        return (database[:,t_index]).cpu().data.numpy()
    if word=='uyyy':
        Hy =  H_grad[:,y_index].reshape(-1,1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        return Hyyy.cpu().data.numpy()
    if word=='(uux)t':
        result=torch.autograd.grad(outputs=(u*Hx).sum(), inputs=database, create_graph=True)[0][:,t_index].reshape(-1, 1)
        return result.cpu().data.numpy()
    if word=='(uux)xx':
        temp=torch.autograd.grad(outputs=(u*Hx).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result=torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        return result.cpu().data.numpy()
    if word=='(u^3)xx':
        temp=torch.autograd.grad(outputs=(u**3).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result=torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        return result.cpu().data.numpy()

    if word == '(u^4)xx':
        temp = torch.autograd.grad(outputs=(u ** 4).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result = torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        return result.cpu().data.numpy()

    if word=='(u(u^2)xx)xx':
        temp = torch.autograd.grad(outputs=(u * Hx).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        temp_1 = torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        temp_2=torch.autograd.grad(outputs=(u*temp_1).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result = torch.autograd.grad(outputs=(temp_2).sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        return result.cpu().data.numpy()


def calculate_terms_PINN(word,Net,database,variables):
    if 'x' in variables:
        x_index=variables.index('x')
    if 't' in variables:
        t_index=variables.index('t')
    if 'y' in variables:
        y_index=variables.index('y')
    if 'z' in variables:
        z_index=variables.index('z')

    u = Net(database)
    H_grad = torch.autograd.grad(outputs=u.sum(), inputs=database, create_graph=True)[0]
    Hx = H_grad[:, x_index].reshape(-1,1)
    if 't' in variables:
        Ht = H_grad[:, t_index].reshape(-1,1)
    if word=='ut':
        return Ht
    if word=='u':
        return u
    if word=='ux':
        return Hx
    if word=='uxx':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return Hxx
    if word=='uxxt':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        Hxxt=torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, t_index]
        return Hxxt
    if word=='ut^2':
        return (Ht**2)
    if word=='uxt':
        result=torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, t_index]
        return result
    if word=='ux^2':
        return (Hx**2)
    if word=='utt':
        utt=torch.autograd.grad(outputs=Ht.sum(), inputs=database, create_graph=True)[0][:, t_index]
        return utt
    if word=='uxxxx':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        Hxxxx = torch.autograd.grad(outputs=Hxxx.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return Hxxxx
    if word=='(u^2)xx':
        H2x= torch.autograd.grad(outputs=(Hx**2).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        H2xx = torch.autograd.grad(outputs=H2x.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return H2xx
    if word=='(uux)x':
        HHx_x=torch.autograd.grad(outputs=(u*Hx).sum(), inputs=database, create_graph=True)[0][:, x_index]
        return HHx_x
    if word=='uxxtt':
        Hxx= torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        Hxxt=torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, t_index].reshape(-1,1)
        Hxxtt=torch.autograd.grad(outputs=Hxxt.sum(), inputs=database, create_graph=True)[0][:, t_index]
        return Hxxtt
    if word=='(u^4)xx':
        H4x=torch.autograd.grad(outputs=(u**4).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        H4xx=torch.autograd.grad(outputs=H4x.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return H4xx
    if word=='(u^3)x':
        H3x = torch.autograd.grad(outputs=(u ** 3).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        return H3x
    if word=='uxxx':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1,1)
        Hxxx= torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, x_index]
        return Hxxx
    if word=='u^3':
        return (u**3)
    if word=='x':
        return database[:,x_index]
    if word=='u^2':
        return (u**2)
    if word=='(1/u)xx':
        temp=torch.autograd.grad(outputs=(u**(-1)).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1,1)
        result=torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:, x_index]
        return result
    if word=='(u^-2*ux)x':
        result= torch.autograd.grad(outputs=(u**(-2)*Hx).sum(), inputs=database, create_graph=True)[0][:, x_index]
        return result
    if word=='ux^2':
        return (Hx**2)
    if word=='uxxxxx':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hxxxx = torch.autograd.grad(outputs=Hxxx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        Hxxxxx = torch.autograd.grad(outputs=Hxxxx.sum(), inputs=database, create_graph=True)[0][:, x_index]
        del Hxxxx
        return Hxxxxx
    if word=='uyy':
        Hy= H_grad[:, y_index].reshape(-1,1)
        Hyy=torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index]
        return (Hyy)
    if word=='ut^3':
        return (Ht**3)
    if word=='sqrt(u)':
        return (torch.sqrt(u))
    if word=='sin(u)':
        return torch.sin(u)
    if word=='sinh(u)':
        return torch.sinh(u)
    if word=='BiLaplace(u)':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        Hxxxx = torch.autograd.grad(outputs=Hxxx.sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        Hy = H_grad[:,  y_index].reshape(-1,1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyyy = torch.autograd.grad(outputs=Hyyy.sum(), inputs=database, create_graph=True)[0][:,y_index].reshape(-1, 1)
        # if 'z' in variables:
        #     Hz =H_grad[:, z_index].reshape(-1,1)
        #     Hzz = torch.autograd.grad(outputs=Hz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        #     Hzzz = torch.autograd.grad(outputs=Hzz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        #     Hzzzz = torch.autograd.grad(outputs=Hzzz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        #     Hxxy=torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        #     Hxxyy=torch.autograd.grad(outputs=Hxxy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        #     Hyyz= torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:,  z_index].reshape(-1, 1)
        #     Hyyzz= torch.autograd.grad(outputs=Hyyz.sum(), inputs=database, create_graph=True)[0][:,  z_index].reshape(-1, 1)
        #     Hxxz=torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:,  z_index].reshape(-1, 1)
        #     Hxxzz=torch.autograd.grad(outputs=Hxxz.sum(), inputs=database, create_graph=True)[0][:,  z_index].reshape(-1, 1)
        #     result=Hxxxx+Hyyyy+Hzzzz+2*Hxxyy+2*Hyyzz+2*Hxxzz
        # else:
        Hxxy = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1,
                                                                                                                 1)
        Hxxyy = torch.autograd.grad(outputs=Hxxy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(
            -1, 1)
        result = Hxxxx + Hyyyy + 2 * Hxxyy
        del Hxxxx
        del Hyyyy
        del Hxxyy
        return result
    if word=='uyyyy':
        Hy = H_grad[:, y_index].reshape(-1, 1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyyy = torch.autograd.grad(outputs=Hyyy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1,
                                                                                                                   1)
        return Hyyyy
    if word=='uxxyy':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hxxy = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hxxyy = torch.autograd.grad(outputs=Hxxy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1,
                                                                                                                   1)
        return Hxxyy
    if word=='uzzzz':
        Hz = H_grad[:, z_index].reshape(-1, 1)
        Hzz = torch.autograd.grad(outputs=Hz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        Hzzz = torch.autograd.grad(outputs=Hzz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        Hzzzz = torch.autograd.grad(outputs=Hzzz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        return Hzzzz
    if word=='uyyzz':
        Hy = H_grad[:, y_index].reshape(-1, 1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyz = torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        Hyyzz = torch.autograd.grad(outputs=Hyyz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1,
                                                                                                                   1)
        return Hyyzz
    if word=='uxxzz':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hxxz = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        Hxxzz = torch.autograd.grad(outputs=Hxxz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1,
                                                                                                                   1)
        return Hxxzz
    if word=='y':
        return database[:,y_index]
    if word=='uy':
        Hy =  H_grad[:, y_index].reshape(-1,1)
        return Hy
    if word=='x^2':
        return (database[:,x_index]**2)
    if word=='y^2':
        return (database[:, y_index]**2)
    if word=='uy^2':
        Hy =  H_grad[:, y_index].reshape(-1,1)
        return (Hy**2)
    if word=='uxy':
        result= torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        return result
    if word=='uz':
        Hz =H_grad[:, z_index].reshape(-1,1)
        return Hz
    if word=='Div(u)':
        Hy = H_grad[:, y_index].reshape(-1, 1)
        if 'z' not in variables:
            result = Hx + Hy
        else:
            Hz =H_grad[:, z_index].reshape(-1,1)
            result=Hx+Hy+Hz
        return result
    if word=='Laplace(u)':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        Hy = H_grad[:, y_index].reshape(-1,1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        if 'z' in variables:
            Hz =H_grad[:, z_index].reshape(-1,1)
            Hzz = torch.autograd.grad(outputs=Hz.sum(), inputs=database, create_graph=True)[0][:,z_index].reshape(-1, 1)
            result = Hxx + Hyy + Hzz
        else:
            result = Hxx + Hyy
        return result
    if word=='uzz':
        Hz = H_grad[:, z_index].reshape(-1, 1)
        Hzz = torch.autograd.grad(outputs=Hz.sum(), inputs=database, create_graph=True)[0][:, z_index].reshape(-1, 1)
        return Hzz
    if word=='Laplace(utt)':
        utt = torch.autograd.grad(outputs=Ht.sum(), inputs=database, create_graph=True)[0][:, t_index]
        uttx= torch.autograd.grad(outputs=utt.sum(), inputs=database, create_graph=True)[0][:, x_index]
        uttxx = torch.autograd.grad(outputs=uttx.sum(), inputs=database, create_graph=True)[0][:,x_index]
        utty = torch.autograd.grad(outputs=utt.sum(), inputs=database, create_graph=True)[0][:, y_index]
        uttyy = torch.autograd.grad(outputs=utty.sum(), inputs=database, create_graph=True)[0][:, y_index]
        if 'z' in variables:
            uttz = torch.autograd.grad(outputs=utt.sum(), inputs=database, create_graph=True)[0][:, z_index]
            uttzz = torch.autograd.grad(outputs=uttz.sum(), inputs=database, create_graph=True)[0][:, z_index]
            result=uttxx+uttyy+uttzz
        else:
            result=uttxx+uttyy
        return result
    if word=='ut^3':
        return (Ht**3)
    if word=='(x+y)':
        return (database[:,x_index]+database[:,y_index])
    if word=='exp(x)':
        return torch.exp(database[:,x_index])
    if word=='uyyt':
        Hy = H_grad[:, 2].reshape(-1,1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyt=torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        return Hyyt
    if word=='sint':
        return torch.sin(database[:,x_index])
    if word=='sinx':
        return torch.sin(database[:, x_index])
    if word=='(uxx+ux/x)^2':
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result=(Hxx+(1/database[:,x_index].reshape(-1,1))*Hx)**2
        return  result
    if word=='x^4':
        return (database[:, x_index]**4)
    if word=='sqrt(x)':
        return (torch.sqrt(database[:, x_index]))
    if word=='exp(-y)':
        return (torch.exp(-database[:, y_index]))
    if word=='t':
        return (database[:,t_index])
    if word=='uyyy':
        Hy =  H_grad[:,y_index].reshape(-1,1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, y_index].reshape(-1, 1)
        return Hyyy
    if word=='(uux)t':
        result=torch.autograd.grad(outputs=(u*Hx).sum(), inputs=database, create_graph=True)[0][:,t_index].reshape(-1, 1)
        return result
    if word=='(uux)xx':
        temp=torch.autograd.grad(outputs=(u*Hx).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result=torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        return result
    if word=='(u^3)xx':
        temp=torch.autograd.grad(outputs=(u**3).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result=torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        return result

    if word == '(u^4)xx':
        temp = torch.autograd.grad(outputs=(u ** 4).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result = torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        return result

    if word=='(u(u^2)xx)xx':
        temp = torch.autograd.grad(outputs=(u * Hx).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        temp_1 = torch.autograd.grad(outputs=(temp).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        temp_2=torch.autograd.grad(outputs=(u*temp_1).sum(), inputs=database, create_graph=True)[0][:, x_index].reshape(-1, 1)
        result = torch.autograd.grad(outputs=(temp_2).sum(), inputs=database, create_graph=True)[0][:,x_index].reshape(-1, 1)
        return result