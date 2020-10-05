% �]�m��l�Ѽ�
clear ; close all; clc

input_layer_size  = 400;  % 20x20 �¥� 0~9 ���Ӥ�
hidden_layer_size = 25;   % �����h25�� hidden units
num_labels = 10;          % 10��label,from 1 to 10 (0���Ϥ��ڭ̰O��10) 

%% 
% ���J�˥��ƾ�(�w����X�MY)
fprintf('Loading and Visualizing Data ...\n')
load('data.mat');
m = size(X, 1);

% �H�����X100�Ӽ˥��Ϥ�(��ı��)
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

%% 
% ���ü˥��ƾڪ����ǫ�A����training set �M test set(�ǲߪ���Ƹ�n�w�������)
temp_1=[X,Y];
temp_2(1:size(temp_1,1),:)=temp_1(randperm(size(temp_1,1)),:);

% �o��Otest set
X_test = temp_2(4001:5000,1:400); 
Y_test = temp_2(4001:5000,401);

% �o��Otraining set
X_train = temp_2(1:4000,1:400);
Y_train = temp_2(1:4000,401);

%% 
%�H����l��parameters(�@�}�ltheta����0�|�X���D)

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

%%

% gradient descent(��פU��)
lambda = 0.1;       %regularization���Y��(�i�վ�)
alpha = 0.3;        %learning rate(�i�վ�)

for i = 1:500       %(��פU��500���A�i�վ�A500���j���ǽT��90%�A�òq�j���O10%)
    
    %forward propagation + backwardpropagation
    %�D�XTheta1,Theta2�b�ثe���w��Theta���J���ײv
    [J, Theta1_grad, Theta2_grad] = cost_grad(Theta1,Theta2, num_labels, X_train, Y_train, lambda);
    
    %�L�XJ����
    fprintf('Value of cost function: %f\n', J);
    
    %gradient descent
    Theta1 = Theta1 - alpha*Theta1_grad;
    Theta2 = Theta2 - alpha*Theta2_grad;
    
    %�w��
    pred_train = predict(Theta1, Theta2, X_train);
    pred_test = predict(Theta1, Theta2, X_test);

    fprintf('Training Set Accuracy: %f\n', mean(double(pred_train == Y_train)) * 100);
    fprintf('Test Set Accuracy: %f\n\n', mean(double(pred_test == Y_test)) * 100);
    
end
    
save Theta Theta1 Theta2  %�N�V�m�n��Theta�x�s�_��
%% 
%�w���A�i�H��X,Y�令�A�Q�n�w�����˥�(��pX_test, Y_test)
pred_train = predict(Theta1, Theta2, X_train);
fprintf('Accuracy: %f\n', mean(double(pred_train == Y_train)) * 100);



