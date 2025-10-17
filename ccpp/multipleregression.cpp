// https://rosettacode.org/wiki/Multiple_regression#C++
// modified after though!!

#include <array>
#include <iostream>
#include <ostream>

static inline void require(_In_ const bool& condition, _In_ const std::string& message) {
    if (condition) return;
    throw std::runtime_error(message);
}

template<typename _TyValue, typename _TyChar, size_t _n_elements>
static inline std::basic_ostream<_TyChar>& operator<<(
    _Inout_ std::basic_ostream<_TyChar>& ostr, _In_ const std::array<_TyValue, _n_elements>& array
) {
    ostr << static_cast<_TyChar>('[');
    for (typename std::array<_TyValue, _n_elements>::const_iterator it = array.cbegin(); it != array.cend(); ++it) ostr << *it;
    return ostr << static_cast<_TyChar>(']');
}

template<size_t _row_count, size_t _column_count> class Matrix final {
    private:
        std::array<std::array<double, _column_count>, _row_count> data;

    public:
        Matrix() : data {} {
            // empty
        }

        Matrix(std::initializer_list<std::initializer_list<double>> values) {
            size_t rp = 0;
            for (auto row : values) {
                size_t cp = 0;
                for (auto col : row) {
                    data[rp][cp] = col;
                    cp++;
                }
                rp++;
            }
        }

        inline double __stdcall get(size_t row, size_t col) const noexcept { return data[row][col]; }

        void __stdcall set(size_t row, size_t col, double value) noexcept { data[row][col] = value; }

        inline std::array<double, _column_count> __stdcall get(size_t row) noexcept { return data[row]; }

        inline void __stdcall set(size_t row, const std::array<double, _column_count>& values) noexcept {
            std::copy(values.begin(), values.end(), data[row].begin());
        }

        template<size_t D> Matrix<_row_count, D> operator*(const Matrix<_column_count, D>& rhs) const noexcept {
            Matrix<_row_count, D> result;
            for (size_t i = 0; i < _row_count; i++) {
                for (size_t j = 0; j < D; j++) {
                    for (size_t k = 0; k < _column_count; k++) {
                        double prod = get(i, k) * rhs.get(k, j);
                        result.set(i, j, result.get(i, j) + prod);
                    }
                }
            }
            return result;
        }

        Matrix<_column_count, _row_count> transpose() const noexcept {
            Matrix<_column_count, _row_count> trans;
            for (size_t i = 0; i < _row_count; i++)
                for (size_t j = 0; j < _column_count; j++) trans.set(j, i, data[i][j]);
            return trans;
        }

        inline void __stdcall toReducedRowEchelonForm() noexcept {
            size_t lead = 0;
            for (size_t r = 0; r < _row_count; r++) {
                if (_column_count <= lead) return;
                auto i = r;

                while (get(i, lead) == 0.0) {
                    i++;
                    if (_row_count == i) {
                        i = r;
                        lead++;
                        if (_column_count == lead) return;
                    }
                }

                auto temp = get(i);
                set(i, get(r));
                set(r, temp);

                if (get(r, lead) != 0.0) {
                    auto div = get(r, lead);
                    for (size_t j = 0; j < _column_count; j++) set(r, j, get(r, j) / div);
                }

                for (size_t k = 0; k < _row_count; k++) {
                    if (k != r) {
                        auto mult = get(k, lead);
                        for (size_t j = 0; j < _column_count; j++) {
                            auto prod = get(r, j) * mult;
                            set(k, j, get(k, j) - prod);
                        }
                    }
                }

                lead++;
            }
        }

        Matrix<_row_count, _row_count> inverse() noexcept requires(_row_count == _column_count) {
            static_assert(_row_count == _column_count, "Not a square matrix!"); // redundant???

            Matrix<_row_count, 2 * _row_count> aug;
            for (size_t i = 0; i < _row_count; i++) {
                for (size_t j = 0; j < _row_count; j++) aug.set(i, j, get(i, j));
                // augment identify matrix to right
                aug.set(i, i + _row_count, 1.0);
            }

            aug.toReducedRowEchelonForm();

            // remove identity matrix to left
            Matrix<_row_count, _row_count> inv;
            for (size_t i = 0; i < _row_count; i++)
                for (size_t j = _row_count; j < 2 * _row_count; j++) inv.set(i, j - _row_count, aug.get(i, j));
            return inv;
        }

        template<size_t RC, size_t CC> friend std::ostream& operator<<(std::ostream&, const Matrix<RC, CC>&);
};

template<size_t RC, size_t CC> std::ostream& operator<<(std::ostream& os, const Matrix<RC, CC>& m) {
    for (size_t i = 0; i < RC; i++) {
        os << '[';
        for (size_t j = 0; j < CC; j++) {
            if (j > 0) os << ", ";
            os << m.get(i, j);
        }
        os << "]\n";
    }

    return os;
}

template<size_t RC, size_t CC> std::array<double, RC> multiple_regression(const std::array<double, CC>& y, const Matrix<RC, CC>& x) {
    Matrix<1, CC> tm;
    tm.set(0, y);

    auto cy = tm.transpose();
    auto cx = x.transpose();
    return ((x * cx).inverse() * x * cy).transpose().get(0);
}

void case1() {
    std::array<double, 5> y { 1.0, 2.0, 3.0, 4.0, 5.0 };
    Matrix<1, 5>          x {
                 { 2.0, 1.0, 3.0, 4.0, 5.0 }
    };
    auto v = multiple_regression(y, x);
    std::cout << v << '\n';
}

void case2() {
    std::array<double, 3> y { 3.0, 4.0, 5.0 };
    Matrix<2, 3>          x {
                 { 1.0, 2.0, 1.0 },
                 { 1.0, 1.0, 2.0 }
    };
    auto v = multiple_regression(y, x);
    std::cout << v << '\n';
}

void case3() {
    std::array<double, 15> y { 52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46 };
    std::array<double, 15> a { 1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83 };

    Matrix<3, 15> x;
    for (size_t i = 0; i < 15; i++) x.set(0, i, 1.0);
    x.set(1, a);
    for (size_t i = 0; i < 15; i++) x.set(2, i, a[i] * a[i]);

    auto v = multiple_regression(y, x);
    std::cout << v << '\n';
}

int main() {
    case1();
    case2();
    case3();

    return 0;
}
