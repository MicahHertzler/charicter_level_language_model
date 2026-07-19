import torch
import matplotlib.pyplot as plt

#makes more of texts you give it
#example baby names

#each predicts next char from string
words=open('names.txt','r').read().splitlines()

#"issabella" i is a likely to start a word, s is likely to fallow i ...

#bigram language model 



b={}
for w in words[:]:
    chs = ['.'] + list(w) + ['.']
    
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1,ch2)
        b[bigram] = b.get(bigram,0) + 1
        #print(ch1, ch2)
print(sorted(b.items(), key = lambda kv: -kv[1]))

N = torch.zeros((27,27),dtype=torch.int32)




chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

print("sttoi \n", stoi)

for w in words: 
    chs = ['.'] + list(w) + ['.']
    
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


plt.figure(figsize=(16,16))
plt.imshow(N, cmap="Blues")
for i in range(N.shape[0]):
    for j in range(N.shape[1]):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color ='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top",color='gray')
plt.axis('off')

plt.show()


print(N[0])

p = N[0].float()

p = p/p.sum()

print(p)

g = torch.Generator().manual_seed(2147483647)


p = (N+1).float() #+1 for model smothing
P = p / p.sum(1,keepdim=True)

for i in range(5):
    out = []
    ix =0
    while True:
        p = P[ix]
     
        ix = torch.multinomial(p, num_samples =1, replacement= True, generator= g).item()
        
        out.append(itos[ix])
        
        if ix == 0:
            break
    print(''.join(out))
    
log_likelihood =0   
count =0 
for w in ["micah"]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logProb=torch.log(prob)
        log_likelihood += logProb
        count += 1
        print(f'{ch1}{ch2}:{logProb:.4f}')
#print("log_likelihood:",log_likelihood)

nnl = -log_likelihood #negative log likely hood is a very nice loss function
print(f'{nnl/count}')

#goal: maximize likelyhood of the data w . r .t model parameters (satistical modeling)
# get nn to tune model parameters to maximize likelyhood 


xs , ys = [], []
for w in words[:]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)


xs = torch.tensor(xs)
ys = torch.tensor(ys)

import torch.nn.functional as F

W = torch.randn((27,27), generator=g,requires_grad=True)

#forward pass 
for k in range(100):
    xecoded = F.one_hot(xs,num_classes=27).float() #input into network: one-hot encoding

    logits = (xecoded @ W) # logcounts
    counts = logits.exp() #eqivelent to N matrix/array 
    probs   = counts/counts.sum(1, keepdims = True) 


    loss = -probs[range(len(ys)), ys].log().mean() 

    #back pass 
   
    loss.backward()

    #update 
    W.data += -0.1 * W.grad
    
    W.grad.zero_() # set to zero the gradient
    
print(loss)