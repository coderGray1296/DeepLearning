[d1,d2,d3,d4,d5,d6,d7,d8,d9,d9,d10,output] = textread('train_new.txt','%f%f%f%f%f%f%f%f%f%f%f','delimiter',136);
x_data = [d1,d2,d3,d4,d5,d6,d7,d8,d9,d9,d10]
[input,minI,maxI] = premnmx(x_data)
inputNums=10
outputNums=1
[xNorm, xps] = mapminmax(x_data); 
[yNorm, yps] = mapminmax(output);
net = netff(xNorm,yNorm)


net=newff(xNorm, yNorm,5,{'tansig','tansig'},'traingdm');
net.trainParam.show = 50;
net.trainParam.lr = 0.01;
net.trainParam.mc = 0.9;
net.trainParam.epochs =5000;
net.trainParam.goal = 1e-3;
[net,tr,res,E]=train(net,xNorm,yNorm);

%res = sim(net,xNorm); %�����
revRes = mapminmax('reverse',res, yps);
%  ����ƽ��������
relErr = abs(output - revRes)./output;
totalRelErr = sum(relErr);
disp(totalRelErr);
len = length(relErr);
%disp(len);
avgRelErr = totalRelErr*100/len;
disp(avgRelErr);
avgRelErr = 0;
iterNums = 0;
trainTime = 0;