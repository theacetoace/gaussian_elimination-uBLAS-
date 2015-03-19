#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/gaussian_elimination.hpp>
#include "gaussian_elimination.hpp"
#include "gtest/gtest.h"
#include <random>
#include <ctime>

namespace ublas = boost::numeric::ublas;

const double EPS = 1e-8;

TEST(Gaussian_Elimination, matrix3x3v1) {
	// test_matrix
	ublas::matrix<double> m(3, 4);
	m(0, 0) =  2; m(0, 1) =  1; m(0, 2) = -1; m(0, 3) =   8;
	m(1, 0) = -3; m(1, 1) = -1; m(1, 2) =  2; m(1, 3) = -11;
	m(2, 0) = -2; m(2, 1) =  1; m(2, 2) =  2; m(2, 3) =  -3;
	// test_answer
	ublas::matrix<double> v(3, 1);
	v(0, 0) =  2;
	v(1, 0) =  3;
	v(2, 0) = -1;

	int res = ublas::gaussian_elimination(m);
	EXPECT_EQ(res, 0);

	ublas::matrix_column<ublas::matrix<double> > col1(m, 3);
	ublas::matrix_column<ublas::matrix<double> > col2(v, 0);
	double diff = 0;
	for (unsigned i = 0; i < m.size1(); ++i) {
		diff += std::abs(col1(i) - col2(i));
	}
	EXPECT_LE(diff, EPS);
	std::cout << m << std::endl;
}

TEST(Gaussian_Elimination, matrix2x2v1) {
	// test_matrix
	ublas::matrix<double> m(2, 3);
	m(0, 0) = 1; m(0, 1) = -1; m(0, 2) = -5;
	m(1, 0) = 2; m(1, 1) =  1; m(1, 2) = -7;
	// test_answer
	ublas::matrix<double> v(2, 1);
	v(0, 0) = -4;
	v(1, 0) =  1;

	int res = ublas::gaussian_elimination(m);
	EXPECT_EQ(res, 0);

	ublas::matrix_column<ublas::matrix<double> > col1(m, 2);
	ublas::matrix_column<ublas::matrix<double> > col2(v, 0);
	double diff = 0;
	for (unsigned i = 0; i < m.size1(); ++i) {
		diff += std::abs(col1(i) - col2(i));
	}
	EXPECT_LE(diff, EPS);
	std::cout << m << std::endl;
}

TEST(Gaussian_Elimination, matrix3x3v2) {
	// test_matrix
	ublas::matrix<double> m(3, 4);
	m(0, 0) = 3; m(0, 1) =  2; m(0, 2) = -5; m(0, 3) = -1;
	m(1, 0) = 2; m(1, 1) = -1; m(1, 2) =  3; m(1, 3) = 13;
	m(2, 0) = 1; m(2, 1) =  2; m(2, 2) = -1; m(2, 3) =  9;
	// test_answer
	ublas::matrix<double> v(3, 1);
	v(0, 0) = 3;
	v(1, 0) = 5;
	v(2, 0) = 4;
	
	int res = ublas::gaussian_elimination(m);
	EXPECT_EQ(res, 0);

	ublas::matrix_column<ublas::matrix<double> > col1(m, 3);
	ublas::matrix_column<ublas::matrix<double> > col2(v, 0);
	double diff = 0;
	for (unsigned i = 0; i < m.size1(); ++i) {
		diff += std::abs(col1(i) - col2(i));
	}
	EXPECT_LE(diff, EPS);
	std::cout << m << std::endl;
}

TEST(Gaussian_Elimination, matrix3x3v3) {
	// test_matrix
	ublas::matrix<double> m(3, 4);
	m(0, 0) = 4; m(0, 1) = 2; m(0, 2) = -1; m(0, 3) = 1;
	m(1, 0) = 5; m(1, 1) = 3; m(1, 2) = -2; m(1, 3) = 2;
	m(2, 0) = 3; m(2, 1) = 2; m(2, 2) = -3; m(2, 3) = 0;
	// test_answer
	ublas::matrix<double> v(3, 1);
	v(0, 0) = -1;
	v(1, 0) =  3;
	v(2, 0) =  1;
	
	int res = ublas::gaussian_elimination(m);
	EXPECT_EQ(res, 0);

	ublas::matrix_column<ublas::matrix<double> > col1(m, 3);
	ublas::matrix_column<ublas::matrix<double> > col2(v, 0);
	double diff = 0;
	for (unsigned i = 0; i < m.size1(); ++i) {
		diff += std::abs(col1(i) - col2(i));
	}
	EXPECT_LE(diff, EPS);
	std::cout << m << std::endl;
}

TEST(Gaussian_Elimination, matrix4x4noanswer) {
	// test_matrix
	ublas::matrix<double> m(3, 4);
	m(0, 0) = 7; m(0, 1) = -2; m(0, 2) = -1; m(0, 3) = 2;
	m(1, 0) = 6; m(1, 1) = -4; m(1, 2) = -5; m(1, 3) = 3;
	m(2, 0) = 1; m(2, 1) =  2; m(2, 2) =  4; m(2, 3) = 5;
	
	int res = ublas::gaussian_elimination(m);
	EXPECT_EQ(res, -1);

	std::cout << m << std::endl;
}

TEST(Gaussian_Elimination, matrix4x4infanswer) {
	// test_matrix
	ublas::matrix<double> m(4, 5);
	m(0, 0) = 2; m(0, 1) =  3; m(0, 2) = -1; m(0, 3) =  1; m(0, 4) = 1;
	m(1, 0) = 8; m(1, 1) = 12; m(1, 2) = -9; m(1, 3) =  8; m(1, 4) = 3;
	m(2, 0) = 4; m(2, 1) =  6; m(2, 2) =  3; m(2, 3) = -2; m(2, 4) = 3;
	m(3, 0) = 2; m(3, 1) =  3; m(3, 2) =  9; m(3, 3) = -7; m(3, 4) = 3;
	
	int res = ublas::gaussian_elimination(m);
	EXPECT_EQ(res, 1);

	std::cout << m << std::endl;
}

TEST(Gaussian_Elimination, matrix4x4numstability) {
	std::srand(time(0));
	// test_matrix
	ublas::matrix<double> m(4, 5);
	m(0, 0) = 2; m(0, 1) =  5; m(0, 2) = 4; m(0, 3) = 1; m(0, 4) = 20;
	m(1, 0) = 1; m(1, 1) =  3; m(1, 2) = 2; m(1, 3) = 1; m(1, 4) = 11;
	m(2, 0) = 2; m(2, 1) = 10; m(2, 2) = 9; m(2, 3) = 7; m(2, 4) = 40;
	m(3, 0) = 3; m(3, 1) =  8; m(3, 2) = 9; m(3, 3) = 2; m(3, 4) = 37;
	
	int res = ublas::gaussian_elimination(m);
	EXPECT_EQ(res, 0);

	ublas::matrix<double> mnoise(m);
	for (unsigned i = 0; i < mnoise.size1(); ++i)
		mnoise(i, 4) += (static_cast<double>(std::rand()) / RAND_MAX - 0.5) / 5;

	res = ublas::gaussian_elimination(mnoise);
	EXPECT_EQ(res, 0);

	double diff = 0;
	for (unsigned i = 0; i < mnoise.size1(); ++i)
		diff += std::abs(mnoise(i, 4) - m(i, 4));

	EXPECT_LE(diff, 0.4);
	std::cout << diff << std::endl;
	std::cout << m << std::endl;
}