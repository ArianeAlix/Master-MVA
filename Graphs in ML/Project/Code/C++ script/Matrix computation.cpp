#include "manage_csv.h"
#include "screen.h"


using namespace std;




float **  compute_sim_connections(float ** A, float ** S_drug, int n1, int n2) {

	float ** S_td;
	S_td = new float*[n1];
	for (int i = 0; i < n1; i++) {
		S_td[i] = new float[n1]();
	}

	cout << endl << "Computing similarity matrix..." << endl;

	int t = 0;
	for (int i = 0; i < n1; ++i)
	{ // We iterate over the disease in the index, that are not in rows filled with 0

		if (t % 100 == 0 || t == n1-1) {
			cout << round((float)t / (float)n1 * 100);
			cout << '%' << '\r';

	
		}
		t = t + 1;

		for (int j = 0; j < n1; ++j) 
		{
			int coeff_norm = 0;

			for (int l = 0; l < n2; ++l) 
			{
				if (A[i][l]!=0)
				{
					for (int k = 0; k < n2; ++k) 
					{
						if (A[j][k] != 0) {
							// Numerator
							S_td[i][j] = S_td[i][j] + A[i][l] * A[j][k] * S_drug[l][k];

							// Coeff to normalize : sum of max possible connections : denominator
							coeff_norm = coeff_norm + A[i][l] * A[j][k];
							
						}
					}
				}
			}
			if (coeff_norm != 0){

				S_td[i][j] = S_td[i][j] / coeff_norm;
			}
		}
	}
	return S_td;
}


float sum(float * A, int n2){
	float  sum = 0;
	for (int j = 0; j < n2; j++) {
		sum += A[j];
	}
	return sum;
}


float sum_col(float ** A, int n1, int col) {
	float  sum = 0;
	for (int i = 0; i < n1; i++) {
		sum += A[i][col];
	}
	return sum;
}




float **  compute_weight_matrix(float ** A, float ** S, int n1, int n2, float lambd) {

	float ** W;
	W = new float*[n1];
	for (int i = 0; i < n1; i++) {
		W[i] = new float[n1]();
	}

	cout << endl << "Computing weight matrix with lambda="<<lambd << "..." << endl;

	int t = 0;
	for (int i = 0; i < n1; ++i)
	{ // We iterate over the disease in the index, that are not in rows filled with 0

		if (t % 100 == 0 || t == n1 - 1) {
			cout << round((float)t / (float)n1 * 100);
			cout << '%' << '\r';

		}
		t = t + 1;

		float order_i = sum(A[i],n2); //order of disease i;

		for (int j = 0; j < n1; ++j)
		{
			float order_j = sum(A[j],n2); //order of disease j

			for (int l = 0; l < n2; ++l)
			{
				if (A[i][l] != 0 && A[j][l] != 0) {
					float order_l = sum_col(A, n1, l); //order of drug l

					W[i][j] += A[i][l] * A[j][l];

					if (order_j != 0) {
						W[i][j] /= order_l;
					}
				}
				
			}
			if (order_i != 0 && order_j != 0) {
				W[i][j] *= S[i][j] / (pow(order_i, (1.0 - lambd)) * pow(order_j, lambd));
			}
		}
	}

	return W;
}



int main()
{
	cout << "Do you want to compute similarity matrices? (0: None , 1: Cross-validation ones, 2: Final one)";
	int yn;
	std::cin >> yn;

	if (yn == 2) {

		int n_drugs = 2115;
		int n_diseases = 1465;


		// Loading the drug similarity table
		float ** S_drug = read_csv("./saved_tables/save_drug_sim_norm.csv", n_drugs, n_drugs);


		// Loading the disease-drug adjacency table
		float ** A = read_csv("./saved_tables/save_A.csv", n_diseases, n_drugs);


		// Computation of similarity matrix with connections via similar drugs
		float **  S_td = compute_sim_connections(A, S_drug, n_diseases, n_drugs);


		// Storing the result line by line in a .csv
		// Creating an object of CSVWriter
		CSVWriter writer("./saved_tables/save_connection_sim.csv");


		for (int row = 0; row < n_diseases; ++row)
		{
			float* arr = S_td[row];
			writer.addDatainRow(arr, arr + n_diseases);
		}



		// Deleting the arrays after the computation is done 

		for (int i = 0; i < n_drugs; i++)
			delete[] S_drug[i];
		delete[] S_drug;


		for (int i = 0; i < n_diseases; i++)
			delete[] A[i];
		delete[] A;


		for (int i = 0; i < n_diseases; i++)
			delete[] S_td[i];
		delete[] S_td;
	}
	

	if (yn == 1) {
		int n_drugs = 2115;
		int n_diseases = 1465;

		// Loading the disease-drug adjacency table
		cout << "Loading " << "./saved_tables/save_drug_sim_norm.csv" << endl;
		float ** S_drug = read_csv("./saved_tables/save_drug_sim_norm.csv", n_drugs, n_drugs);


		// Computing weight matrices for all A tables generated for cross-validation by removing 20% of the interactions
		string path_A_tables = "./A_tables/A_train_";


		// Computing S_td matrices
		string path_Std_tables = "./Std_tables/S_td_";

		for (int a = 0; a < 10; a++) {
			string path_A = path_A_tables + to_string(a+1) + ".csv";

			string path_Std = path_Std_tables + to_string(a+1) + ".csv";

			// Loading each A
			cout << "Loading " << path_A << endl;
			float ** A= read_csv(path_A, n_diseases, n_drugs);

			// Computing the S_td table
			float ** S_td = compute_sim_connections(A, S_drug, n_diseases, n_drugs);


			// Saving the result to a .csv file 
			CSVWriter writer(path_Std);

			for (int row = 0; row < n_diseases; ++row)
			{
				float* arr = S_td[row];
				writer.addDatainRow(arr, arr + n_diseases);
			}

			cout << path_Std << " saved." << endl << endl;




			// Deleting the array after all the weight matrices are computed for this turn
			for (int i = 0; i < n_diseases; i++)
				delete[] S_td[i];
			delete[] S_td;

			for (int i = 0; i < n_diseases; i++)
				delete[] A[i];
			delete[] A;


			}
		// Deleting the array after all the weight matrices are computed
		for (int i = 0; i < n_diseases; i++)
			delete[] S_drug[i];
		delete[] S_drug;
		}


	char yon;

	cout << "Do you want to compute the weight matrices? (y/n)";

	std::cin >> yon;

	if (yon == 'y' || yon == 'Y') {
		int n_drugs = 2115;
		int n_diseases = 1465;

		string set;

		cout << "For which training set? (1-10 or final)";

		std::cin >> set;

		float ** A;
		float ** S;
		string path_S_tables;
		string path_W_tables;

		if (set == "final") {
			// Loading the disease-drug adjacency table
			A = read_csv("./saved_tables/save_A.csv", n_diseases, n_drugs);


			// Computing weight matrices for all S tables generated with 9 different alphas
			path_S_tables = "./S_final_tables/S_";


			// Computing weight matrices with different lambdas (for all S tables)
			path_W_tables = "./W_final_tables/W_";

		}
		else {
			// Loading the disease-drug adjacency table
			A = read_csv("./saved_tables/save_A.csv", n_diseases, n_drugs);


			// Computing weight matrices for S of the ith set
			path_S_tables = "./S_tables/S_" + set + "_" ;


			// Computing weight matrices with different lambdas (for all S tables)
			path_W_tables = "./W_tables/W_" + set + "_";

		}

		float alphas[9] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
		float lambdas[9] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };

		for (int a = 0; a < 9; a++) {
			string alpha = to_string(alphas[a]).substr(0,4);
			string path_S = path_S_tables + alpha + ".csv";

			cout << "Loading " << path_S << endl;
			// Loading the final similarity table
			float ** S = read_csv(path_S, n_diseases, n_diseases);

			for (int l = 0; l < 9; l++) {
				float lambd = lambdas[l];

				// Computation of the weight matrix
				float ** W = compute_weight_matrix(A, S, n_diseases, n_drugs, lambd);

				// Storing the result line by line in a .csv
				// Creating an object of CSVWriter
				string lambda = to_string(lambdas[l]).substr(0, 4);
				string path_W = path_W_tables + alpha + "_" +lambda +".csv";

				CSVWriter writer(path_W);

				for (int row = 0; row < n_diseases; ++row)
				{
					float* arr = W[row];
					writer.addDatainRow(arr, arr + n_diseases);
				}

				cout << path_W << " saved." << endl;

				// Deleting the arrays after the computation for this lambda is done
				for (int i = 0; i < n_diseases; i++)
					delete[] W[i];
				delete[] W;


			}
			// Deleting the array after all the weight matrices are computed for this alpha
			for (int i = 0; i < n_diseases; i++)
				delete[] S[i];
			delete[] S;

		}


		// Deleting the arrays after the computation is done 

		for (int i = 0; i < n_diseases; i++)
			delete[] A[i];
		delete[] A;


	}




	std::cout << "Press any key to exit...";
	std::cin.get();
	std::cin.get();





	return 0;



}






