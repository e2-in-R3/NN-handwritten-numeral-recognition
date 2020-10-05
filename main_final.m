% 設置初始參數
clear ; close all; clc

input_layer_size  = 400;  % 20x20 黑白 0~9 的照片
hidden_layer_size = 25;   % 中間層25個 hidden units
num_labels = 10;          % 10個label,from 1 to 10 (0的圖片我們記為10) 

%% 
% 載入樣本數據(已知的X和Y)
fprintf('Loading and Visualizing Data ...\n')
load('data.mat');
m = size(X, 1);

% 隨機取出100個樣本圖片(視覺化)
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

%% 
% 打亂樣本數據的順序後，分成training set 和 test set(學習的資料跟要預測的資料)
temp_1=[X,Y];
temp_2(1:size(temp_1,1),:)=temp_1(randperm(size(temp_1,1)),:);

% 這邊是test set
X_test = temp_2(4001:5000,1:400); 
Y_test = temp_2(4001:5000,401);

% 這邊是training set
X_train = temp_2(1:4000,1:400);
Y_train = temp_2(1:4000,401);

%% 
%隨機初始化parameters(一開始theta全為0會出問題)

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

%%

% gradient descent(梯度下降)
lambda = 0.1;       %regularization的係數(可調整)
alpha = 0.3;        %learning rate(可調整)

for i = 1:500       %(梯度下降500次，可調整，500次大概準確度90%，亂猜大約是10%)
    
    %forward propagation + backwardpropagation
    %求出Theta1,Theta2在目前給定的Theta對於J的斜率
    [J, Theta1_grad, Theta2_grad] = cost_grad(Theta1,Theta2, num_labels, X_train, Y_train, lambda);
    
    %印出J的值
    fprintf('Value of cost function: %f\n', J);
    
    %gradient descent
    Theta1 = Theta1 - alpha*Theta1_grad;
    Theta2 = Theta2 - alpha*Theta2_grad;
    
    %預測
    pred_train = predict(Theta1, Theta2, X_train);
    pred_test = predict(Theta1, Theta2, X_test);

    fprintf('Training Set Accuracy: %f\n', mean(double(pred_train == Y_train)) * 100);
    fprintf('Test Set Accuracy: %f\n\n', mean(double(pred_test == Y_test)) * 100);
    
end
    
save Theta Theta1 Theta2  %將訓練好的Theta儲存起來
%% 
%預測，可以把X,Y改成你想要預測的樣本(比如X_test, Y_test)
pred_train = predict(Theta1, Theta2, X_train);
fprintf('Accuracy: %f\n', mean(double(pred_train == Y_train)) * 100);



