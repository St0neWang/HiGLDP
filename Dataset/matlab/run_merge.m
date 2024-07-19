maxiter = 20;
restartProb = 0.50;

lnc = {'Sim_mat_Lnc_mRNA', 'Sim_mat_Lnc_Protein', 'Sim_mat_Lnc_RBP', 'Sim_mat_Lnc_Lnc'};
disease = {'Sim_mat_DO_DO', 'Sim_mat_DO_circRNA', 'Sim_mat_DO_Metabolite'};

Lnc_vector = merge(lnc, restartProb, maxiter);
dlmwrite(['../feature/Lnc_vector.txt'], Lnc_vector, '\t');

DO_vector = merge(disease, restartProb, maxiter);
dlmwrite(['../feature/DO_vector.txt'], DO_vector, '\t');
