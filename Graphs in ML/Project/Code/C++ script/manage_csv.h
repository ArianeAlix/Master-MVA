#pragma once


#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>
#include <set>
#include <string>
#include <algorithm>


using namespace std;


/*
* A class to create and write data in a csv file.
*/
class CSVWriter
{
	std::string fileName;
	std::string delimeter;
	int linesCount;

public:
	CSVWriter(std::string filename, std::string delm = ",") :
		fileName(filename), delimeter(delm), linesCount(0)
	{}
	/*
	* Member function to store a range as comma seperated value
	*/
	template<typename T>
	void addDatainRow(T first, T last);
};

/*
* This Function accepts a range and appends all the elements in the range
* to the last row, seperated by delimeter (Default is comma)
*/
template<typename T>
void CSVWriter::addDatainRow(T first, T last)
{
	std::fstream file;
	// Open the file in truncate mode if first line else in Append Mode
	file.open(fileName, std::ios::out | (linesCount ? std::ios::app : std::ios::trunc));

	// Iterate over the range and add each lement to file seperated by delimeter.
	for (; first != last; )
	{
		file << *first;
		if (++first != last)
			file << delimeter;
	}
	file << "\n";
	linesCount++;

	// Close the file
	file.close();
}




float ** read_csv(string path, int n1, int n2) {

	float ** tab;
	tab = new float*[n1];
	for (int i = 0; i < n1; i++) {
		tab[i] = new float[n2];
	}

	std::ifstream file(path);

	cout << "Loading table... ";

	for (int row = 0; row < n1; ++row)
	{
		std::string line;
		std::getline(file, line);
		if (!file.good())
			break;

		std::stringstream iss(line);

		for (int col = 0; col < n2; ++col)
		{
			std::string val;


			std::getline(iss, val, ',');

			if (!iss.good())
				break;

			std::stringstream convertor(val);
			convertor >> tab[row][col];
		}
	}

	cout << "Done." << endl;

	return tab;

}
