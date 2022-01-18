import torch
import os
import fnmatch

def print_with_write(log_file,log_str):
    log_file.write(log_str+'\n')
    log_file.flush()
    print(log_str)

def save_model(state,save_model_path,modelname):
    filename = os.path.join(save_model_path,modelname+'_'+str(state['epoch']+1)+'.pth')
    torch.save(state,filename)

def load_model(Net, optimizer, model_file,log_file):
    assert os.path.exists(model_file),'There is no model file from'+model_file
    print_with_write(log_file,'load from '+model_file)
    checkpoint = torch.load(model_file)
    Net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']+1
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return Net, optimizer, start_epoch


# you can modify this function and use it.  not use it in default
def load_the_newest_model(save_model_path,log_file):
    model_file_name = []
    all_file_name = os.listdir(os.path.join('./',save_model_path))
    for file_name in all_file_name:
        if(fnmatch.fnmatch(file_name,'*.ckpt')):
            model_file_name.append(os.path.join(save_model_path,file_name))
    assert len(model_file_name) > 0,'there is no *.ckpt file'
    model_file = max(model_file_name,key=os.path.getmtime)
    print_with_write(log_file,'load from '+model_file)
    checkpoint = (torch.load(model_file))
    return checkpoint