def decode(preds,lens,lstm_decoder,alpha):
    _,batch_size,nclass=preds.size()
    zero_cat=torch.FloatTensor(1,1,nclass-1).fill_(0)
    a=time.time()
    y_seq=[""]*batch_size
    yi_index_old=93984
    loop_count=0
    for batch_index in range(batch_size):
      y, hid1,hid2 = lstm_decoder.decode(Variable(zero_cat).cuda())
      pred=preds[:,batch_index,:]
      for xi_index in range(lens[batch_index]):
         xi=pred[xi_index][:]
         flag=True
         loop_count=0
         while flag:
           loop_count+=1
           y_i=y[0][0][:]
           ytu = nn.functional.softmax(y_i+xi,dim=0)
           max_value,yi=torch.max(ytu,0)
           yi_index=int(yi.data)
           if loop_count>1 and yi_index==yi_index_old:
              flag=False
              break
           if yi_index!=0:
              yi_index_old=yi_index
              t=torch.IntTensor([yi_index-1])
              l=torch.IntTensor([1])
              t_onehot=utils.oneHot(t,l,nclass-1)
              t_onehot=t_onehot.permute(1,0,2) #btn
              y, hid1,hid2 = lstm_decoder.decode(Variable(t_onehot).cuda(),hid1,hid2)
              y_seq[batch_index]+=alpha[yi_index-1]
           else:
              flag=False
    return y_seq
