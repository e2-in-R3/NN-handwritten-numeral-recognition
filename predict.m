function p = predict(Theta1, Theta2, X)
    %�o�Ө�ƬO�Ψӧ�Xy_hat�ȳ̤j�����@��
    m = size(X, 1);
    h1 = sigmoid([ones(m, 1) X] * Theta1);
    h2 = sigmoid([ones(m, 1) h1] * Theta2);
    [dummy, p] = max(h2, [], 2);
end
