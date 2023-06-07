masks = struct();
masks.normal_mask_lambda = @(mat) cos(2*pi*mat/size(mat,1) + pi*size(mat,1)/4) ...
    .*cos(2*pi*mat/size(mat,2) + pi*size(mat,2)/4);

masks.log_mask_lambda = @(mat) log(mat/size(mat,1)) .* log(mat/size(mat,2));

masks.first_block_lambda = @(mat) kron([1 0;0 0], mat*ones(size(mat,1),size(mat,2)));
masks.second_block_lambda = @(mat) kron([0 1;0 0], mat*ones(size(mat,1),size(mat,2)));
masks.third_block_lambda = @(mat) kron([0 0;0 1], mat*ones(size(mat,1),size(mat,2)));
masks.fourth_block_lambda = @(mat) kron([0 0;1 0], mat*ones(size(mat,1),size(mat,2)));

mat_4x4     = meshgrid(0:3,0:3);
mat_8x8     = meshgrid(0:7,0:7);
mat_16x16   = meshgrid(0:15,0:15);

for mask = {'normal_mask_lambda', 'log_mask_lambda', 'first_block_lambda', ...
    'second_block_lambda', 'third_block_lambda', 'fourth_block_lambda'},
    mask = cell2mat(mask);
    fprintf('Applying 4x4 matrix over %s: \n----\n',mask);
    (masks.(mask))(mat_4x4)
#    fprintf('Applying 8x8 matrix over %s: \n----\n',mask);
#    masks.(mask)(mat_8x8)
#    fprintf('Applying 16x16 matrix over %s: \n----\n',mask); 
#    masks.(mask)(mat_16x16)
    fprintf('\n\n');
endfor

