clear all;
close all;

%%Input Image
input_path = 'hdrs';
name = 'HallOfFame';
%th  = [0.008;8]; %AmikeusBeaverDamPM2
th = [0.01;10]; %HallOfFame

%%PSF
load('psf.mat');

%%Mask 
load('uniform_4e_6stops.mat');
weight_mask = mask;

%%Load image
img = double(hdrread([input_path '/' name '.hdr']));
ldr = zeros(size(img));

%Coding
for ch = 1:3
    %Simulate PSF
    coded = imfilter(img(:,:,ch), K(:,:,ch), 'same', 'conv', 'symmetric');
    %Simulate the modulatation of light arriving at the sensor coded by
    %mask
    coded = weight_mask.*coded;
    %Simulate bracketing. To do: implement a simulation of any
    %bracketing method with reasonable results to make it authomatic
    coded(coded<th(1)) = th(1);
    coded(coded>th(2)) = th(2);
    ldr(:,:,ch) = coded;
end

ldr = (ldr - min(ldr(:))) / (max(ldr(:)) - min(ldr(:)));
%Simulate camera response - gamma coding and quantization
ldr = uint8(ldr.^(1/2.2)*255);
imwrite(ldr,[name '_coded.png']);

%%Reconstruction from coded image
result = zeros(size(img));
for ch = 1:3
    I_observation = ((double(ldr(:,:,ch)))/255).^(2.2);
    Initial = I_observation./weight_mask;
    weight_mask(I_observation>0.95) = 0;
    weight_mask(I_observation<0.01) = 0;
    I_observation(weight_mask==0) = 0;
    MM = zeros(size(weight_mask));
    MM(weight_mask==0) = 1;

    I_init = Interpolation_Initial(Initial,MM);
    I_init(I_init>max(Initial(:))) = max(Initial(:));
    I_init(I_init<min(Initial(:))) = min(Initial(:));

    %Gaussian
    psf = K(:,:,ch);
    psf = psf / sum(psf(:));
    %Local smooth
    k = fspecial('gaussian',[13 13],3*1.591); %Filter from local contrast normalization
    I_init = imfilter(I_init, k, 'same', 'conv', 'symmetric');
    %%Load filters
    kernels = load('filters_ours_obj1.26e+04.mat');
    d = kernels.d;   
    %Ground truth signal - just for comparisons
    signal = img(:,:,ch);
    verbose_admm = 'brief';     
    %%Reconstruction parameters
    max_it = [100]; 
    lambda_residual = 200000.0;
    lambda = 3.0;
    lambda_smooth = 1;

    tic();
    [z, sig_rec_ours, sig_rec_ours_blurred] = admm_solve_conv2D_weighted_sampling_smooth_dirac(I_observation, d, weight_mask, psf, lambda_residual, lambda, lambda_smooth, I_init, max_it, 1e-5, signal, verbose_admm);
    tt = toc;
    result(:,:,ch) = sig_rec_ours;
end

test = (result - min(result(:))) / (max(result(:)) - min(result(:)));
ref = (img - min(img(:))) / (max(img(:)) - min(img(:)));
mse = mean(mean((ref - test).^2, 1), 2);
psnr = mean(10 * log10(1/mse));
hdrwrite(result,sprintf('%s_Lres_%.1f_L_%.1f_Ls_%.1f_%.2fdB.hdr',name,lambda_residual,lambda,lambda_smooth,psnr));

