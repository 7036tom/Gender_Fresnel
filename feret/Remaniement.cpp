
#include <iostream>
#include <string>
#include <fstream> 
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
	int nbelements = 965;
	int Dim = 28;
	int* Liste = new int[(Dim*Dim+1)*nbelements];
	int random;
	int intermediaire = 0;
	
	string File_name = "2828BaseTest";
	ifstream Rfile("2828BaseTest", ios::in);  // Open in read mode

	        if(Rfile)  // if no error
	        {
	        	//int i = 0;
	        	
	        	for(size_t i = 0; i < nbelements; i++)
	        	{

	        		for(size_t j  = 0; j < Dim*Dim+1; j++)
	        		{
	        			Rfile >> Liste[j+i*(Dim*Dim+1)];	
	        		}
	        	}       	
	        }
	        else
	                cerr << "The file can't be opened " << endl;
	
	//cout << Liste[Dim*Dim-1]<<endl;
	//cout << Liste[Dim*Dim]<<endl;
	
	/*
	for(size_t i = 0; i < 2599; i++) //786?
	{
	    	
	        for(size_t j  = 0; j < Dim*Dim+1; j++)
	        	{
	        		if (j < Dim*Dim)
	        		{
	        			cout << Liste[j+i*(Dim*Dim+1)]<<" ";	
	        		}
	        		else if (j == Dim*Dim)
	        		{
	        			cout << Liste[j+i*(Dim*Dim+1)];
	        		}
	        	}
	        cout << endl;
	   } */
	    
	
	//Mélangeons la liste pour que hommes et femmes se cotoient

	for (size_t z = 0; z < 5000; z++)
	{
		for (size_t i = 0; i < nbelements; i++)
		{
			random = rand() % nbelements;
			for (size_t j = 0; j < Dim*Dim+1;j++)
			{
				intermediaire = Liste[j+random*(Dim*Dim+1)];
				Liste[j+random*(Dim*Dim+1)] = Liste[j+i*(Dim*Dim+1)];
				Liste[j+i*(Dim*Dim+1)] = intermediaire;
			}
		}
	}

	for(size_t i = 0; i < nbelements; i++)
	    {
	        for(size_t j  = 0; j < Dim*Dim+1; j++)
	        	{
	        		if (j < Dim*Dim)
	        		{
	        			cout << Liste[j+i*(Dim*Dim+1)]<<" ";	
	        		}
	        		else
	        		{
	        			cout << Liste[j+i*(Dim*Dim+1)];
	        		}
	        	}
	        cout << endl;
	    } 
	
	    



	
	delete [] Liste;
	return (0);
}