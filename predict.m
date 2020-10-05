function p = predict(Theta1, Theta2, X)
    %這個函數是用來找出y_hat值最大的那一個
    m = size(X, 1);
    h1 = sigmoid([ones(m, 1) X] * Theta1);
    h2 = sigmoid([ones(m, 1) h1] * Theta2);
    [dummy, p] = max(h2, [], 2);
end
