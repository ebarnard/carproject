qp = importdata('curvature_qp.csv');

n = size(qp.data, 1);

x = qp.data(:,1);
y = qp.data(:,2);
dx = qp.data(:,3);
dy = qp.data(:,4);
l = qp.data(:,5);
u = qp.data(:,6);
f = qp.data(:,7);
H = qp.data(:,8:end);
w = u * 2;

options = optimoptions('quadprog',...
    'Algorithm','interior-point-convex',...
    'OptimalityTolerance',5e-16,...
    'Display','iter');

[alpha_quadprog, fval, exitflag] = quadprog(H, f', [], [], [], [], l, u, [], options);

if exitflag ~= 1
   error('Could not opt');
end

m = osqp;
A = eye(n);
m.setup(H, f, A, l, u, 'max_iter', 20000, 'eps_abs', 1e-10, 'eps_rel', 1e-10);
results = m.solve();
alpha_osqp = results.x;

% Generate new coordinates
x_quadprog = x + alpha_quadprog.*dx;
y_quadprog = y + alpha_quadprog.*dy;

x_osqp = x + alpha_osqp.*dx;
y_osqp = y + alpha_osqp.*dy;

% Plot results

outer_x = x + 0.5*w.*dx;
outer_y = y + 0.5*w.*dy;

inner_x = x - 0.5*w.*dx;
inner_y = y - 0.5*w.*dy;

% Complete the track loops

x(n+1) = x(1);
y(n+1) = y(1);

x_quadprog(n+1) = x_quadprog(1);
y_quadprog(n+1) = y_quadprog(1);

outer_x(n+1) = outer_x(1);
outer_y(n+1) = outer_y(1);

inner_x(n+1) = inner_x(1);
inner_y(n+1) = inner_y(1);

figure(2);
clf
hold on
plot(outer_x, outer_y, 'g');
plot(inner_x, inner_y, 'g');
plot(x, y, 'b');
plot(x_quadprog, y_quadprog, 'r');
plot(x_osqp, y_osqp, 'black');
axis equal;
