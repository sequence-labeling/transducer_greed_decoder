def decode(preds,lens,lstm_decoder,alpha):
    _,batch_size,nclass=preds.size()
    zero_cat=torch.FloatTensor(1,1,nclass-1).fill_(0)
    a=time.time()
    y_seq=[""]*batch_size
    loop_count=0
    for batch_index in range(batch_size):
      y, hid1,hid2 = lstm_decoder.decode(Variable(zero_cat).cuda())
      pred=preds[:,batch_index,:]
      for xi_index in range(lens[batch_index]):
         xi=pred[xi_index][:]
         loop_count=0
         while True:
           loop_count+=1
           y_i=y[0][0][:]
           ytu = y_i+xi
           max_value,yi=torch.max(ytu,0)
           yi_index=int(yi.data)
           if yi_index==0 or loop_count>4:
              break
           if yi_index!=0:
              t=torch.IntTensor([yi_index-1])
              l=torch.IntTensor([1])
              t_onehot=utils.oneHot(t,l,nclass-1)
              t_onehot=t_onehot.permute(1,0,2) #btn
              y, hid1,hid2 = lstm_decoder.decode(Variable(t_onehot).cuda(),hid1,hid2)
              y_seq[batch_index]+=alpha[yi_index-1]

