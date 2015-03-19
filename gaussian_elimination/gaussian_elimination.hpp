#pragma once

#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace boost {
	namespace numeric {
		namespace ublas {
			/* returns 
			-1 if there is no solution
			0 if solution exists and the last column of the matrix is the answer
			1 if there are infinite number of solutions 
			*/
			template <typename T>
			int gaussian_elimination(ublas::matrix<T> &equation) {
				/// constants
				const T EPS = 1e-8;
				const unsigned iterations = std::min(equation.size1(), equation.size2() - 1);

				/// processing top-down elimination
				for (unsigned i = 0; i < iterations; ++i) {
					/// finding max val from lefted rows
					int max_val_pos = i;
					for (unsigned j = i + 1; j < equation.size1(); ++j) {
						max_val_pos = std::abs(equation(j, i)) > std::abs(equation(max_val_pos, i)) ? j : max_val_pos;
					}
					ublas::matrix_row< ublas::matrix<T> > row1(equation, i);
					ublas::matrix_row< ublas::matrix<T> > row2(equation, max_val_pos);
					row1.swap(row2);

					if (std::abs(equation(i, i)) <= EPS) { continue; }
					
					for (unsigned j = i + 1; j < equation.size1(); ++j) {
						T diff = equation(j, i) / equation(i, i);
						ublas::row(equation, j) -= ublas::row(equation, i) * diff;
					}
				}

				/// processing bottom-up elimination
				for (int i = static_cast<int>(equation.size1()) - 1; i >= 0; --i) {
					if (std::abs(equation(i, i)) < EPS) {
						return std::abs(equation(i, equation.size2() - 1)) < EPS ? 1 : -1;
					}

					/// processing normalizing
					T normalize = 1.0 / equation(i, i);
					ublas::matrix_row< ublas::matrix<T> > crow(equation, i);
					crow *= normalize;

					for (int j = i - 1; j >= 0; --j) {
						T diff = equation(j, i) / equation(i, i);
						ublas::row(equation, j) -= ublas::row(equation, i) * diff;
					}
				}
				
				return 0;
			}
		}
	}
}