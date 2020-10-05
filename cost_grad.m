function [J, Theta1_grad, Theta2_grad] = cost_grad(Theta1,Theta2, num_labels, X, Y, lambda)
    %先找具體函數                               
    m = size(X, 1); 
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));
    %forward propagation + backward propagation
    a_0 = [ones(m, 1), X];
    z_1 = a_0 * Theta1;
    a_0 = [ones(m, 1), sigmoid(z_1)];
    z_1 = a_0 * Theta2;
    a_1 = sigmoid(z_1);

    eye_matrix = eye(num_labels);
    Y_matrix = eye_matrix(Y,:);

    J = (1/m)*sum(sum((-Y_matrix).*log(a_1)-(1-Y_matrix).*(log(1-a_1))));

    theta1 = Theta1(1:end, 2:end);
    theta2 = Theta2(1:end, 2:end);
    J = J+sum((lambda/(2*m))*(sum(sum(theta1.^2))+sum(sum(theta2.^2))));
for t=1:m
    a_0 = X(t,:);
    a_0 = [1, a_0];
    z_1 = a_0*Theta1;
    a_1 = sigmoid(z_1);
    a_1 = [1, a_1];
    z_2 = a_1*Theta2;
    a_2 = sigmoid(z_2);
   
    partial_Lz2 = (a_2) - (Y_matrix(t,:)); %1*10
    z_1 = [1, z_1];
    partial_Lz1 = (partial_Lz2*Theta2').*sigmoidGradient(z_1); %?
    partial_Lz1 = partial_Lz1(2:end);
    
    Theta2_grad = Theta2_grad + a_1' * partial_Lz2; %26*10
    Theta1_grad = Theta1_grad + a_0' * partial_Lz1; %401*26  
end
    Theta2_grad = (1/m)*Theta2_grad;
    Theta1_grad = (1/m)*Theta1_grad;

    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end)+(lambda/m)*Theta2(:, 2:end);
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end)+(lambda/m)*Theta1(:, 2:end);
end
