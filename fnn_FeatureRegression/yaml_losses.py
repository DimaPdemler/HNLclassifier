import torch
import torch.nn as nn



def custom_loss_normal(y_pred, y_true):
    se_loss = (y_pred - y_true) ** 2
    # print(se_loss.shape)
    num_features = int(output_dim)

    loss_list=[]

    for i in range(num_features):

        if i in customlossindices:
            RMSE=((y_pred[:,i]-y_true[:,i])/y_true[:,i])**2
            mask = (y_true[:,i] > 1)

            RMSE_meanloss=torch.mean(RMSE[mask])
            MSE_meanloss = torch.mean(se_loss[:,i][~mask])

            se_loss[mask, i] = RMSE[mask]

            loss_list.append(MSE_meanloss.item())
            loss_list.append(RMSE_meanloss.item())
        else:
            loss = torch.mean(se_loss[:,i])
            loss_list.append(loss.item())
    
    full_loss = torch.mean(se_loss)
    return loss_list, full_loss

def custom_loss_no_mse(y_pred, y_true):
    se_loss = (y_pred - y_true) ** 2
    MSE_loss = se_loss.clone()
    
    num_features = int(output_dim)
    loss_list = []
    total_loss = 0

    for i in range(num_features):
        if i in customlossindices:
            RMSE = ((y_pred[:, i] - y_true[:, i]) / y_true[:, i]) ** 2
            mask = (y_true[:, i] > 1)
            
            RMSE_meanloss = torch.mean(RMSE[mask])
            MSE_meanloss = torch.mean(se_loss[:, i][~mask])

            MSE_loss[:, i] = 0  
            MSE_loss[mask, i] = RMSE[mask]

            loss_list.append(MSE_meanloss.item())
            loss_list.append(RMSE_meanloss.item())

            total_loss += RMSE_meanloss  
        else:
            loss = torch.mean(se_loss[:, i])
            loss_list.append(loss.item())
            total_loss += loss  

    full_loss = total_loss / num_features 
    return loss_list, full_loss