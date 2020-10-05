close all

load('Theta.mat');                %�ޤJ�w�V�m�n��Theta
I = imread('1.png');              %�ޤJ�Ϥ�(���ɦW)
J = imresize(I, [20, 20]);        %�N�Ϥ�����20*20����
K = rgb2gray(J);                  %�N�m��Ϥ��令�Ƕ��Ϥ�
L = 1 - im2double(K);             %�N�¥զ�˹L��
M = L(:)';                        %�N20*20�Ԧ�1*40
displayData(M);                   %�L�X�Ϥ�
h1 = sigmoid([1, M] * Theta1);    %�N�Ϥ��z�LTheta1,Theta2�e��1*10�����G
h2 = sigmoid([1, h1] * Theta2);
[dummy, p] = max(h2, [], 2);      %��X10�Ӥ��̤j��
fprintf('%d\n', p);               %�L�X���G