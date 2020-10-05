function W = randInitializeWeights(L_in, L_out)

epsilon_init = 0.12;
W = rand(1 + L_in, L_out) * 2 * epsilon_init - epsilon_init;

end
