import torch
import numpy as np
import argparse
import time
import util
from pipeline import train_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--data',type=str,default='/data/pems_data/pems_vldb/short_term',help='data path')
parser.add_argument('--adjdata',type=str,default='/data/pems_data/pems_vldb/adj_speed.npy',help='adj data path')
parser.add_argument('--seq_num',type=int,default=12,help='')
parser.add_argument('--hid_dim',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=3,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=9638,help='number of nodes')
parser.add_argument('--tau',type=int,default=0.25,help='temperature coefficient')
parser.add_argument('--random_feature_dim',type=int,default=64,help='random feature dimension')
parser.add_argument('--node_emb_dim',type=int,default=32,help='node embedding dimension')
parser.add_argument('--time_emb_dim',type=int,default=32,help='time embedding dimension')
parser.add_argument('--use_residual', action='store_true', help='use residual connection')
parser.add_argument('--use_bn', action='store_true', help='use batch normalization')
parser.add_argument('--use_spatial', action='store_true', help='use spatial loss')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.0001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--save',type=str,default='checkpoint/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

def main():
    device = torch.device(args.device)
    adj_mx = util.load_adj(args.adjdata)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, args.seq_num)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    
    edge_indices = torch.nonzero(supports[0] > 0)

    print(args)

    trainer = train_pipeline(scaler, args.seq_num, args.in_dim, args.hid_dim, args.num_nodes, args.tau,
                             args.random_feature_dim, args.node_emb_dim, args.time_emb_dim, args.use_residual,
                             args.use_bn, args.use_spatial, args.dropout, args.learning_rate, args.weight_decay,
                             device, supports, edge_indices)

    print("start training...",flush=True)
    his_loss =[]
    test_time = []
    val_time = []
    train_time = []
    
    for i in range(1, args.epochs+1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device) # (B, T, N, F)
            trainx = trainx.transpose(1, 2) # (B, N, T, F)
            trainy = torch.Tensor(y).to(device) # (B, T, N, F)
            trainy = trainy.transpose(1, 2) # (B, N, T, F)
            metrics = trainer.train(trainx, trainy[:,:,:,0])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            t2 = time.time()
            train_time.append(t2-t1)
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 2)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 2)
            metrics = trainer.eval(testx, testy[:,:,:,0])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Validation Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        
        log = 'Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        print(log.format(mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse), flush=True)

    # test
    test_loss = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': []}
    test_mape = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': []}
    test_rmse = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': []}
    s1 = time.time()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 2)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 2)
        metrics = trainer.eval(testx, testy[:,:,:,0], flag='horizon')
        for k in range(12):
            test_loss[str(k)].append(metrics[0][k])
            test_mape[str(k)].append(metrics[1][k])
            test_rmse[str(k)].append(metrics[2][k])
    s2 = time.time()
    log = 'Epoch: {:03d}, Test Inference Time: {:.4f} secs'
    print(log.format(i,(s2-s1)))
    test_time.append(s2-s1)
    amae = []
    amape = []
    armse = []
    for k in range(12):
        amae.append(np.mean(test_loss[str(k)]))
        amape.append(np.mean(test_mape[str(k)]))
        armse.append(np.mean(test_rmse[str(k)]))
        log = 'Model performance for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(k+1, amae[-1], amape[-1], armse[-1]))

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
