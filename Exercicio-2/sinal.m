intervalo = [-1,1];
lbanda = [0 0.01];

sinal = idinput(1200,'prbs',lbanda,intervalo);
sinal = iddata([],sinal,0.1);

plot(sinal);
grid on;

saida = sinal.u(:);

csvwrite('dados001.csv',saida);