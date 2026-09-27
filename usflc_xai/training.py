import time, torch, sys, math
from torch import save, load
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd

def train_step(encoder, pretrained_model, loss, optimizer, data_loader, batch_size, perform_eval, device):
    
    # train encoder #
    
    total_s = len(data_loader)
    
    l = 0
    n = 0
    crt = 0
    encoder.train()
    
    for s, batch in enumerate(data_loader):
        
        start= time.time()
        
        _, y, h, loss_fn = forward_backward_prop(encoder=encoder,
                                                 pretrained_model=pretrained_model,
                                                 loss=loss,
                                                 optimizer=optimizer,
                                                 batch=batch,
                                                 device=device, 
                                                 mode="backpropagation")
        
        crt += perform_eval(prediction=h, ground_truth=y, mode="count")
        
        l += loss_fn.item() * y.size(0)
        n += y.size(0)
        
        time_diff = time.time()-start
        
        #print(('Iteration {}: {:.2f} secs; train loss: {:.4f}; train acc: {:.4f}').format(s, time_diff, l / n, crt / n),
        #      end="\r", file=sys.stdout, flush=True)
        
    return l / n, crt / n



def valid_step(encoder, pretrained_model, loss, optimizer, data_loader, batch_size, perform_eval, device):
    
    l = 0
    n = 0
    crt = 0
    
    with torch.no_grad():
    
        encoder.eval()

        for batch in data_loader:
            
            _, y, h, loss_fn = forward_backward_prop(encoder=encoder,
                                                     pretrained_model=pretrained_model,
                                                     loss=loss,
                                                     optimizer=optimizer,
                                                     batch=batch,
                                                     device=device, 
                                                     mode="forwardpropagation")
            #loss_fn = loss(h, y)
            
            crt += perform_eval(prediction=h, ground_truth=y, mode="count")
            
            l += loss_fn.item() * y.size(0)
            n += y.size(0)
                
    return l / n, crt / n

    
def test_step(num_classes, encoder, pretrained_model, loss, optimizer, data_loader, batch_size, perform_eval, device):
     
    l = 0
    n = 0
    crt = 0
    conf_m = 0
    
    with torch.no_grad():
    
        encoder.eval()

        for batch in data_loader:
            
            _, y, h, loss_fn = forward_backward_prop(encoder=encoder,
                                                     pretrained_model=pretrained_model,
                                                     loss=loss,
                                                     optimizer=optimizer,
                                                     batch=batch,
                                                     device=device, 
                                                     mode="forwardpropagation")
            
            crt += perform_eval(prediction=h, ground_truth=y, mode="count")
            conf_m += confusion_matrix(num_classes=num_classes, prediction=h, ground_truth=y)
            
            l += loss_fn.item() * y.size(0)
            n += y.size(0)
            
    return l / n, crt / n, conf_m
    
    
    
def forward_backward_prop(encoder, pretrained_model, loss, optimizer, batch, device, mode="backpropagation"): 
    
    x, b, A, y, train_mask = data_extractor(batch=batch, pretrained_model=pretrained_model, device=device, mode=mode)
    
    h = encoder(x, A, b, train_mask)
    
    loss_fn = loss(h, y)
    
    if(mode=="backpropagation"):
        
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()
    
    return x, y, h, loss_fn
    

def data_extractor(batch, pretrained_model, device, mode):
    
    x=batch.x.to(device)
    A=batch.edge_index.to(device)
    y=torch.tensor(batch.y, dtype=torch.int64).detach().to(device)
    b=batch.batch.to(device)
    train_mask=1

    ## data augmentation ##
    
    if mode=="backpropagation":
        train_mask = batch.train_mask.to(device)
    #    prob = torch.rand(1)
        
    #    if prob > 0:

    #        train_mask = torch.unsqueeze(torch.tensor(choice([0, 1], len(x), p=[0.5, 0.5])), 1).to(device)
    #        ind, _=(train_mask!=0).nonzero(as_tuple=True)

    #        A, _ = subgraph(subset=ind, edge_index=A, edge_attr=None)
    #        x = x*train_mask
            
    #        eta = 1/0.5    
            
    
    return x, b, A, y, train_mask
   
    

def acc(prediction, ground_truth, mode="count"):
    
    _, y_hat = torch.max(prediction.data, dim=1)
    crt = (y_hat == ground_truth.data).sum().item()
    
    if(mode=="proportion"):
        crt = crt / ground_truth.size(0)
        
    return crt

def confusion_matrix(num_classes, prediction, ground_truth):
    
    conf_index = [0, 1]
    
    if num_classes==1 or num_classes==2:
        
        conf_index = [0, 1]
    
    if num_classes==3:
        
        conf_index = [0, 1, 2]
    
    if num_classes==4:
        
        conf_index = [0, 1, 2, 3]
    
    _, y_hat = torch.max(prediction.data, dim=1)
    
    y_hat=y_hat.data.to('cpu').numpy()
    ground_truth = ground_truth.data.to('cpu').numpy()
    
    conf_m = pd.crosstab(y_hat, ground_truth)
    conf_m = conf_m.reindex(index=conf_index, columns=conf_index, fill_value=0)
    conf_m = conf_m.to_numpy().flatten()
    
    return conf_m

def print_result(file_name, results, init="yes"):
    
    if init=="yes":
        
        with open(file_name, 'a') as f:
            
            f.write('data\timg_encoder\tmodel\tnum_layers\ttest_acc\ttrain_acc\n')
            
        f.close()
    
    else:
        
        data_id=results[0]
        pretrained_encoder_id=results[1]
        encoder_id=results[2]
        num_layers=results[3]
        acc_test=results[4]
        acc_train=results[5]
        
        with open(file_name, 'a') as f:
            
            f.write('{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\n'.format(data_id, pretrained_encoder_id, 
                                                              encoder_id, 
                                                              num_layers, acc_test, acc_train))
        f.close()



def print_confusion_matrix(file_name, confusion_matrix, init="yes"):

    conf_m = confusion_matrix.flatten()
    conf_m_l = len(conf_m)
    
    if init=="yes":
        
        with open(file_name, 'a') as f_2:

            for k in range(conf_m_l):

                f_2.write('c{}'.format(k))

                if k < conf_m_l-1:
                    f_2.write('\t')
                else:
                    f_2.write('\n')

        f_2.close()

    else:
        with open(file_name, 'a') as f_2:

            for k in range(conf_m_l):

                f_2.write('{:d}'.format(int(conf_m[k])))

                if k < conf_m_l-1:
                    f_2.write('\t')
                else:
                    f_2.write('\n')

        f_2.close()


def print_prediction(file_name, encoder, data_loader, device, init="yes"):
    
    
        
        if init=="yes":
            
            with open(file_name, 'a') as f_3:
                
                for batch in data_loader:

                    _, _, _, y,_ = data_extractor(batch=batch, pretrained_model=None, device=device, mode="forwardpropagation")

                    for k in range(len(y)):

                        f_3.write('{:d}'.format(y[k].item()))
                        f_3.write('\n')

                #f_3.write('\n')
                
            f_3.close()
            
        else:
            
            with torch.no_grad():

                encoder.eval()
                
                with open(file_name, 'a') as f_3:

                    for batch in data_loader:

                        x, b, A, y, m = data_extractor(batch=batch, pretrained_model=None, device=device, mode="forwardpropagation") 
                        
                        h = encoder(x, A, b, m)
    
                        _, y_hat = torch.max(h.data, dim=1)
                        
                        y_hat = y_hat.data.to('cpu').numpy()
                        
                        for k in range(len(y_hat)):

                            f_3.write('{:d}'.format(y_hat[k].item()))
                            f_3.write('\n')
                       

                    #f_3.write('\n')
                
                f_3.close()     
        

def print_pred_prob(file_name, encoder, data_loader, num_classes, device):

            
            with torch.no_grad():

                encoder.eval()
                
                with open(file_name, 'a') as f_3:

                    for batch in data_loader:

                        x, b, A, y, m = data_extractor(batch=batch, pretrained_model=None, device=device, mode="forwardpropagation") 
                        
                        h = encoder(x, A, b, m)
    
                        #_, y_hat = torch.max(h.data, dim=1)
                        
                        #y_hat = y_hat.data.to('cpu').numpy()
                        
                        for k in range(len(y)):
                            
                            f_3.write('{:d}\t'.format(y[k].item()))
                            
                            for j in range(num_classes):
                                
                                f_3.write('{:.4f}'.format(h[k,j].item()))
                                
                                if j < (num_classes-1):
                                    f_3.write('\t')
                                else:
                                    f_3.write('\n')
                            
                            #f_3.write('\n')

                    #f_3.write('\n')
                
                f_3.close()    

        
def save_results(encoder, optimizer, scheduler, last_epoch, save_patience, dir_name):
            
    file_name=dir_name + 'epoch_' + str(last_epoch) +'.ckpt'
    
    if (last_epoch+1) % save_patience==0:
        save({
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'last_epoch': last_epoch
             },
            f=file_name
        )
    
def save_best_results(encoder, optimizer, scheduler, per_eval_old, per_eval, last_epoch, callback_count, dir_name):
            
    file_name=dir_name + 'best_results.ckpt'
        
    if last_epoch==0:
        
        per_eval_old=per_eval
        
        save({
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'last_epoch': last_epoch
            },
                f=file_name
            ) 
        
    elif last_epoch > 0:
        
        per_diff=per_eval-per_eval_old

        if per_diff < 0:
            save({
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'last_epoch': last_epoch
            },
                f=file_name
            ) 
            
            per_eval_old=per_eval
            callback_count = 0
            
        else:
            
            callback_count += 1
    
    return per_eval_old, callback_count
    
def learning_rate_scheduler(scheduler, adaptive_lr):
    
    if adaptive_lr:
        scheduler.step()
            

            
def backup_fn(backup, encoder, optimizer, scheduler, last_epoch, backup_file_name):
    
    if backup==True:
        
        checkpoint = torch.load(backup_file_name)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['last_epoch']
        
    return encoder, optimizer, scheduler, last_epoch

    
def my_trainer(encoder,
               pretrained_model,
               loss, 
               optimizer, 
               scheduler,
               T_max,
               train_dataset, 
               test_dataset, 
               batch_size, 
               num_epochs, 
               last_epoch,
               adaptive_lr, 
               perform_eval, 
               device,
               callback,
               callback_patience,
               save_dir_name,
               save_patience):
    
    
    # should fill some overhead operations!!#
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    

    # run training procedure #
    
    per_valid_old=0
    callback_count=0
    
    for r in range(num_epochs):
        
        start = time.time()
    
        loss_train, per_train = train_step(encoder=encoder, 
                                           pretrained_model=pretrained_model,
                                           loss=loss,
                                           optimizer=optimizer,
                                           data_loader=train_loader,
                                           batch_size=batch_size,
                                           perform_eval=perform_eval,
                                           device=device)
        
        time_diff = time.time()-start
        
        loss_valid, per_valid = valid_step(encoder=encoder,
                                           pretrained_model=pretrained_model,
                                           loss=loss,
                                           optimizer=optimizer,
                                           data_loader=test_loader,
                                           batch_size=batch_size,
                                           perform_eval=perform_eval,
                                           device=device)
        
        
        # report learning rate #
        
        curr_lr = optimizer.param_groups[0]['lr']
        
        # change learning rate adaptively #
        
        learning_rate_scheduler(scheduler=scheduler, adaptive_lr=adaptive_lr)
        
        # save check points #
        
        save_results(encoder=encoder, 
                     optimizer=optimizer,
                     scheduler=scheduler,
                     last_epoch=r+(last_epoch+1), 
                     save_patience=save_patience, 
                     dir_name=save_dir_name)
        
        # save the best model according to validation loss #
        
        per_valid_old, callback_count=save_best_results(encoder=encoder, 
                                                        optimizer=optimizer,
                                                        scheduler=scheduler,
                                                        per_eval_old=per_valid_old, 
                                                        per_eval=loss_valid,
                                                        last_epoch=r+(last_epoch+1),
                                                        callback_count=callback_count,
                                                        dir_name=save_dir_name)
        
        ### "callback" procedure ###
        
        if (callback==True) and (callback_count==callback_patience):
            
            best_file_name = save_dir_name + 'best_results.ckpt'
            
            encoder, _, _,_ = backup_fn(backup=True, 
                                        encoder=encoder, 
                                        optimizer=optimizer, 
                                        scheduler=scheduler, 
                                        last_epoch=last_epoch, 
                                        backup_file_name=best_file_name)
            if curr_lr > 1e-6:
                curr_lr=0.99*curr_lr
            
            optimizer = torch.optim.Adam(encoder.parameters(), lr=curr_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            
            callback_count=0
        
        
        
        print(('Epoch {}: {:.2f} secs; train loss: {:.4f};'
               ' valid loss: {:.4f}; train acc: {:.4f}; valid acc: {:.4f}; lr: {:.8f}').format(r+(last_epoch+1)+1, 
                                                                                            time_diff, 
                                                                                            loss_train, 
                                                                                            loss_valid, 
                                                                                            per_train, 
                                                                                            per_valid, 
                                                                                            curr_lr), 
              end="\r", file=sys.stdout, flush=True)

    
    # should fill some test procedure!!#
    
    
    return loss_train, loss_valid, per_train, per_valid


def score_stat(prob_sorted, 
               ind, 
               y, 
               num_classes, 
               stat=True, 
               q_hat=0, 
               lam=0,
               k_th=0,
               random=False):
    
    score_i=0
    predset_i=[]
    k=0
    
    for k in range(num_classes):

        if k==0:
            
            score_i += prob_sorted[k].item()
            predset_i.append(ind[k].item())
            
        if stat == True:
            
            if k > 0: 
                
                score_i += prob_sorted[k].item()
                predset_i.append(ind[k].item())
            
            if ind[k].item() == y.item():
                
                score_i += lam*max(0, ((k+1) - k_th))
                
                if random == True:
                    
                    u=torch.rand(1).item()
                    score_i -= u*prob_sorted[k].item()  
                
                
                break
            
        if stat == False:
            
            if k==0:
                
                penalty = lam*max(0, ((k+1) - k_th))
                
                if (score_i + penalty - q_hat) > 0:
                    
                    score_i += penalty
                   
                    break 
                    
            if k > 0: 
 
                score_i += prob_sorted[k].item()
                predset_i.append(ind[k].item())
                penalty = lam*max(0, ((k+1) - k_th))
                

                if (score_i + penalty - q_hat) > 0:
                    
                    if random==False:
                        
                        score_i += penalty
                        
                    
                    if random==True:
                        
                        u=torch.rand(1).item()
                        s_k = prob_sorted[k].item()

                        if (k+1) > k_th:
                            s_k = s_k + lam

                        if (score_i + penalty- u*s_k) - q_hat > 0:

                            score_i -= u*prob_sorted[k].item()
                            predset_i = predset_i[0:k]
                            
                        score_i += penalty
                    
                    break
                      
        
    return score_i, predset_i, len(predset_i)



def confusion_matrix02(num_classes, y_hat, ground_truth):
    
    conf_index = [0, 1]
    
    if num_classes==1 or num_classes==2:
        
        conf_index = [0, 1]
    
    if num_classes==3:
        
        conf_index = [0, 1, 2]
    
    if num_classes==4:
        
        conf_index = [0, 1, 2, 3]
 
    conf_m = pd.crosstab(y_hat, ground_truth)
    conf_m = conf_m.reindex(index=conf_index, columns=conf_index, fill_value=0)
    conf_m = conf_m.to_numpy().flatten()
    
    return conf_m


def conformal_score(num_classes, 
                    encoder, 
                    pretrained_model, 
                    loss, 
                    optimizer, 
                    data_loader, 
                    perform_eval, 
                    alpha,
                    lam,
                    k_th,
                    random,
                    device):
    
    
    score_list = []
    predset_list = []
    predsize_list = []
    
    high = 1 - alpha
    n = 0
    
    with torch.no_grad():

        encoder.eval()

        for batch in data_loader:

            _, y, h, loss_fn = forward_backward_prop(encoder=encoder,
                                                     pretrained_model=pretrained_model,
                                                     loss=loss,
                                                     optimizer=optimizer,
                                                     batch=batch,
                                                     device=device, 
                                                     mode="forwardpropagation")

            n += y.size(0)

            prob = torch.nn.functional.softmax(h, dim=1)
            prob_sorted, ind = torch.sort(prob, dim=1, descending=True)

            for i in range(len(y)):

                score_i, predset_i, predsize_i = perform_eval(prob_sorted[i], 
                                                              ind[i], 
                                                              y[i], 
                                                              num_classes=num_classes, 
                                                              stat=True, 
                                                              q_hat=0,
                                                              lam=lam,
                                                              k_th=k_th,
                                                              random=random)

                score_list.append(score_i)
                predset_list.append(predset_i)
                predsize_list.append(predsize_i)


        #high = math.ceil(high*(n+1))/(n+1)
        #q_hat = torch.quantile(torch.tensor(score_list), high, interpolation='higher').item()
        
        score_list,_ = torch.sort(torch.tensor(score_list), descending=False)
        high = math.ceil((1-alpha)*(n+1))
        q_hat = score_list[high-1].item()
         
    
    return score_list, predset_list, predsize_list, q_hat, high / (n+1)



def conformal_prediction(num_classes, 
                         encoder, 
                         pretrained_model, 
                         loss, 
                         optimizer, 
                         data_loader, 
                         perform_eval, 
                         alpha,
                         lam,
                         k_th,
                         random,
                         q_hat,
                         device):
    
    score_list = []
    predset_list = []
    predsize_list = []
    y_s_list = []
    y_hat_list = []
    y_list = []
    acc_m = 0
    acc_s = 0
    n_s = 0
    n_m = 0
    n = 0
    
    with torch.no_grad():

        encoder.eval()
        
        for batch in data_loader:

            _, y, h, loss_fn = forward_backward_prop(encoder=encoder,
                                                     pretrained_model=pretrained_model,
                                                     loss=loss,
                                                     optimizer=optimizer,
                                                     batch=batch,
                                                     device=device, 
                                                     mode="forwardpropagation")

            n += y.size(0)

            prob = torch.nn.functional.softmax(h, dim=1)
            prob_sorted, ind = torch.sort(prob, dim=1, descending=True)
            
            
            
            for i in range(len(y)):

                score_i, predset_i, predsize_i = score_stat(prob_sorted[i], 
                                                            ind[i], 
                                                            y[i], 
                                                            num_classes=num_classes, 
                                                            stat=False, 
                                                            q_hat=q_hat,
                                                            lam=lam,
                                                            k_th=k_th,
                                                            random=random)

                y_list.append(y[i].item())
                
                if len(predset_i) > 1:
                    n_m += 1
                if len(predset_i)==1:
                    n_s += 1
                    y_hat_list.append(predset_i[0])
                    y_s_list.append(y[i].item())


                if list(set(predset_i).intersection(set([y[i].item()])))!=[]:

                    if len(predset_i) > 1:
                        acc_m += 1
                    if len(predset_i)==1:
                        acc_s += 1

                score_list.append(score_i)
                predset_list.append(predset_i)
                predsize_list.append(predsize_i)


        conf_m = confusion_matrix02(num_classes=num_classes, y_hat=np.array(y_hat_list), ground_truth=np.array(y_s_list))

        return score_list, predset_list, predsize_list, y_list, acc_m, acc_s, n_s, n_m, conf_m





def cover_label(predset_i, num_classes):
    
    coverlabel=[]
    l=0
    
    if len(predset_i)==1:

        l=0
        
        for i in range(0, num_classes):
            
            if(predset_i[0]==i):
                coverlabel=l
                break
                
            l += 1

    if len(predset_i)==2:

        if num_classes==2:
            l=2
        
        if num_classes==3:
            l=3
        
        if num_classes==4:
            l=4
        
        for i in range(0, num_classes):
            for j in range(i+1, num_classes):
                
                if(predset_i[0]==i) and (predset_i[1]==j):
                    coverlabel=l
                    break
                
                l+= 1


    if len(predset_i)==3:
        
        if num_classes==3:
            l=6
        
        if num_classes==4:
            l=10
        
        for i in range(0,num_classes):
            for j in range(i+1, num_classes):
                for k in range(j+1, num_classes):
                    
                    if(predset_i[0]==i) and (predset_i[1]==j) and (predset_i[2]==k):
                        coverlabel=l
                        break
                        
                    l += 1

                    
    if len(predset_i)==4:
         
        if num_classes==4:
            l=14
            coverlabel=l
        
    return coverlabel


def cp_table(y, predset, num_classes):
    
    coverlabel_list=[]
    acc_list=[]
    #max_l=4
    
    if num_classes == 2:
        ind_th = 3
        
    if num_classes == 3:
        ind_th = 7
    
    if num_classes == 4:
        ind_th = 15
    
    for i in range(len(y)):

        predset_i = predset[i]
        predset_i,_ = torch.sort(torch.tensor(predset_i), descending=False)

        predset_i=predset_i.numpy()

        if list(set(predset_i).intersection(set([y[i]])))==[]:

            coverlabel_list.append(cover_label(predset_i, num_classes))
            acc_list.append(0)

        if list(set(predset_i).intersection(set([y[i]])))!=[]:

            coverlabel_list.append(cover_label(predset_i, num_classes))
            acc_list.append(1)

    mydata= pd.DataFrame({"label": coverlabel_list, "acc": acc_list})
    tab = pd.crosstab(mydata.label, mydata.acc)
    tab = tab.reindex(index=[i for i in range(ind_th)], columns=[0,1], fill_value=0)
    results_cp = tab.to_numpy().flatten()

    return results_cp
