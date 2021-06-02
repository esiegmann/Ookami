class RandomState;

class Random {
private:
    const int r;
    const int s;
    const double beta;
    int cur;
    double *u;
    unsigned int simple_state;

    void generate();

    inline unsigned int simple();

    Random operator=(const Random&);
    Random(const Random&);

public:
    Random(unsigned int seed = 5461);
    ~Random();
    inline double get() {
        if (cur >= r) generate();
        return u[cur++];
    };

    void getv(int n, double * RESTRICT v);

    void getbytes(int n, unsigned char * RESTRICT v);
    void getbytes2(int n, unsigned char * RESTRICT v);

    RandomState getstate();

    void setstate(const RandomState &s);
};
