from model import *
from helper import *
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")

modf = relKKT_general(mode = '2',norm=True)
# tar = '/home/lxyang/git/pdqpnet/pkl/8906_valid/QPLIB_8906_17.mps.pkl'
tar = '/home/lxyang/git/pdqpnet/pkl/8845_valid/QPLIB_8845_6.mps.pkl'
tar = '/home/lxyang/git/pdqpnet/pkl/3547_train/QPLIB_3547_143.mps.pkl'
# tar = '/home/lxyang/git/pdqpnet/pkl/8559_valid/QPLIB_8559_17.mps.pkl'
# tar = '/home/lxyang/git/pdqpnet/pkl/3913_valid/QPLIB_3913_256.mps.pkl'
# tar = '/home/lxyang/git/pdqpnet/pkl/8906_valid/QPLIB_8906_459.mps.pkl'


perts=torch.randn(size=(400,))
# perts=perts+[.1]*50
# perts=perts+[.2]*50
# perts=perts+[.3]*50
# perts=[.0001,.001,.01,.1,.2,.3,.4]

records = []
z = torch.tensor(0.0)
x_norm = []
y_norm = []
norms = []
gaps = []
press = []
dress = []
total_loss = []

ori_x,ori_y,sc,pres,dres,gap = sol_check(tar,device,modf)
records.append([sc,pres,dres,gap,z,z])
x_norm.append(z.item())
y_norm.append(z.item())
norms.append(z.item())
gaps.append(gap.item())
press.append(pres.item())
dress.append(dres.item())
total_loss.append(sc.item())

print(ori_x)

for p in perts:
    x,y,sc,pres,dres,gap = sol_check(tar,device,modf,p)
    norm1 = torch.norm(x-ori_x,2)/torch.norm(ori_x,2)
    norm2 = torch.norm(y-ori_y,2)/torch.norm(ori_y,2)
    records.append([sc,pres,dres,gap,norm1,norm2])
    x_norm.append(norm1.item())
    y_norm.append(norm2.item())
    norms.append((norm1+norm2).item())
    gaps.append(gap.item())
    press.append(pres.item())
    dress.append(dres.item())
    total_loss.append(sc.item())



for i in range(len(records)):
    print(round(records[i][0].item(),4),end='  ')
print()
for i in range(len(records)):
    print(round(records[i][1].item(),4),end='  ')
print()
for i in range(len(records)):
    print(round(records[i][2].item(),4),end='  ')
print()
for i in range(len(records)):
    print(round(records[i][3].item(),4),end='  ')
print('\n------------------------------------------------')
for i in range(len(records)):
    print(round(records[i][-1].item(),4),end='  ')
print()
for i in range(len(records)):
    print(round(records[i][-2].item(),4),end='  ')
print()


plt.scatter(norms,total_loss)
# plt.xscale('log')
plt.xlabel('Distance to z*')
plt.ylabel('Total Residual')
plt.savefig('../plots/distance/totalloss.png', bbox_inches='tight')
plt.savefig('../plots/distance/totalloss.pdf', format="pdf", bbox_inches="tight")
plt.clf()

plt.scatter(x_norm,press)
# plt.xscale('log')
plt.xlabel('Distance to z*')
plt.ylabel('Primal Residual')
plt.savefig('../plots/distance/pres.png', bbox_inches='tight')
plt.savefig('../plots/distance/pres.pdf', format="pdf", bbox_inches="tight")
plt.clf()

plt.scatter(y_norm,dress)
# plt.xscale('log')
plt.xlabel('Distance to z*')
plt.ylabel('Dual Residual')
plt.savefig('../plots/distance/dres.png', bbox_inches='tight')
plt.savefig('../plots/distance/dres.pdf', format="pdf", bbox_inches="tight")
plt.clf()

plt.scatter(norms,gaps)
plt.yscale('log')
plt.xlabel('Distance to z*')
plt.ylabel('Primal-dual Gap')
plt.savefig('../plots/distance/gao.png', bbox_inches='tight')
plt.savefig('../plots/distance/gao.pdf', format="pdf", bbox_inches="tight")
plt.clf()