% ME655-WS(BME656-WS) - Virtual Lab Experience 3
% last rev. Oct 2022

clear all
close all
clc

% Create Simulation data
load('Data.mat')
LH_hip_traj = timeseries([Lhip_pos Lhip_vel Lhip_acc Rhip_pos Rhip_vel Rhip_acc],t);

% initial conditions for human controller
Lhip_pos0=Lhip_pos(1);
Lhip_vel0=Lhip_vel(1);
Tau_h_max=30; %[Nm] max torque human can exert

%% Robot motor control

% Oscillator and filter settings:
epsilon=12;
nu=.5;
M=6;
lambda=.95;
N=80;
h=2.5*N;

t_start=5.0; %[s] time of AFO activation
dt_active_control=10; %[s] wait 10s after AFO is active before turning assistance ON

%% CLME BLUE Training

% Only using first 30 seconds to train data
train_len = round(length(Rhip_pos)/2);
Rhip_p = Rhip_pos(1:train_len);
Rhip_v = Rhip_vel(1:train_len);
Rhip_a = Rhip_acc(1:train_len);
Lhip_p = Lhip_pos(1:train_len);
Lhip_v = Lhip_vel(1:train_len);
Lhip_a = Lhip_acc(1:train_len);

% healthy states
phiH = [Rhip_p, Rhip_v];
phiH_dot = [Rhip_v, Rhip_a];
phiH_tot = [phiH; phiH_dot];

% paretic states
phiP = [Lhip_p, Lhip_v];
phiP_dot = [Lhip_v, Lhip_a];
phiP_tot = [phiP; phiP_dot];


% calculating average and standard deviation
phiH_avg = mean(phiH_tot);
phiH_std = std(phiH_tot);
S_H = [phiH_std(1) 0; 0 phiH_std(2)];

phiP_avg = mean(phiP_tot);
phiP_std = std(phiP_tot);
S_P = [phiP_std(1) 0; 0 phiP_std(2)];


% Normalize Data
x_H = zeros(2,train_len);
x_P = zeros(2,train_len);
for i = 1:train_len
    x_H(:,i) = (S_H)\(phiH_tot(i,:) - phiH_avg)';
    x_P(:,i) = (S_P)\(phiP_tot(i,:) - phiP_avg)';
end


% Covariance Matrices
M_hp = zeros(2,2);
M_hh = zeros(2,2);
for i = 1:2
    for j = 1:2
        M_hp(i,j) = sum(x_H(i,:).*x_P(j,:));
        M_hh(i,j) = sum(x_H(i,:).*x_H(j,:));
    end
end


% Calculating Big and Small K
C = ((M_hh)\M_hp)';

big_K = S_P * C / (S_H);
small_k = (-1) * big_K * phiH_avg' + phiP_avg';


% Predicted Training Paretic States
phiP_hat_tot = big_K * phiH_tot' + small_k;
phiP_hat = phiP_hat_tot(1);
phiP_dot_hat = phiP_hat_tot(2);


% Calculating Q and R
T = 60/20134; % sample period = time / # frames
G = [(T^2)/2; T];
Q = G * G' * var(Lhip_a);
R = [var(phiP_dot_hat - Lhip_v) 0; 0 var(phiP_hat - Lhip_p)];
A = [1 T; 0 1];
