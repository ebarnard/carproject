% Min curvature model from http://dx.doi.org/10.1016/j.compstruc.2007.04.028

track = importdata('../Git/rust_controller/data/tracks/big_track_2500.csv');

n = size(track.data, 1);

x = track.data(:,1);
y = track.data(:,2);
w = track.data(:,3);

dx = -(circshift(y, -1) - circshift(y, 1)) / 2;
dy = (circshift(x, -1) - circshift(x, 1)) / 2;
norm = hypot(dx, dy);
dx = diag(dx ./ norm);
dy = diag(dy ./ norm);

% curve is interpolated cubic splines parameterized on t
% A*c = B*x (c is curvature vector, x is x or y position vector)

A = gallery('tridiag',n,1,4,1);
A(1,n) = 1;
A(n,1) = 1;

B = gallery('tridiag',n,6,-12,6);
B(1,n) = 6;
B(n,1) = 6;

A = full(A);
B = full(B);

% c = D*x
D = A\B; %inv(full(A))*full(B);%
D = D .* (abs(D) > 1e-7);
D_2 = D'*D;

% alpha * dx must not go outside the track
alpha_max = 0.5 * w - 0.04;
alpha_min = -0.5 * w + 0.04;

% Must be positive definite as squared value
%if ~all(eig(H) > 0)
%   error('H not positive definite. Optimization not convex.'); 
%end

alpha = zeros(n, 1);

new_x = x;
new_y = y;
new_dx = dx;
new_dy = dy;
alpha_sum = alpha;

for i = 1:4
    D_2_dx = D_2*new_dx;
    D_2_dy = D_2*new_dy;

    H = new_dx'*D_2_dx + new_dy'*D_2_dy;
    B = new_dx*D_2*new_x + new_dy*D_2*new_y;
    B = B';
    
    options = optimoptions('quadprog',...
        'Algorithm','interior-point-convex',...
        'OptimalityTolerance',9e-17,...
        'Display','iter');

    [alpha, fval, exitflag] = quadprog(H, B, [], [], [], [], alpha_min, alpha_max, [], options);

    if exitflag ~= 1
       error('Could not opt');
    end

    new_x = new_x + alpha.*diag(new_dx);
	new_y = new_y + alpha.*diag(new_dy);
    
    alpha_min = alpha_min - alpha;
    alpha_max = alpha_max - alpha;
    alpha_sum = alpha_sum + alpha;
    
    new_xy = interparc(linspace(0, 1, n+1), new_x, new_y, 'csape');
    new_x = new_xy(1:n,1);
    new_y = new_xy(1:n,2);
    
    new_dx = -(circshift(new_y, -1) - circshift(new_y, 1)) / 2;
    new_dy = (circshift(new_x, -1) - circshift(new_x, 1)) / 2;
    norm = hypot(new_dx, new_dy);
    new_dx = diag(new_dx ./ norm);
    new_dy = diag(new_dy ./ norm);
end

% Generate new coordinates
%new_x = new_x + alpha.*diag(dx);
%new_y = new_y + alpha.*diag(dy);

% Calculate point-to-point distance

x_shifted = circshift(x, -1);
y_shifted = circshift(y, -1);

distances2 = hypot(x_shifted - x, y_shifted - y);
sum(distances2)

new_x_shifted = circshift(new_x, -1);
new_y_shifted = circshift(new_y, -1);

distances = hypot(new_x_shifted - new_x, new_y_shifted - new_y);
sum(distances)

% Save coordinates
s = cumsum(distances);
csv = [new_x, new_y, s, alpha_sum];
csvwrite('track1000_minc.csv',csv);

% Plot results

outer_x = x + 0.5*w.*diag(dx);
outer_y = y + 0.5*w.*diag(dy);

inner_x = x - 0.5*w.*diag(dx);
inner_y = y - 0.5*w.*diag(dy);

% Complete the track loops

x(n+1) = x(1);
y(n+1) = y(1);

new_x(n+1) = new_x(1);
new_y(n+1) = new_y(1);

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
plot(new_x, new_y, 'r');
