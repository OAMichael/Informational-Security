#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//функция, котоая проверяет число на простоту
int isPrime(long int n){
        for (long int i = 2; i*i <= n; ++i){
                if (n % i == 0) {
                        return 0;
                }
        }
        return 1;
}

//функция, которая возвращает простое рандомное число длины,
//введенной с клавиатуры и больше
//upd: длина простого числа определяется рандомно
long int lookingForSimple() {

        int n;
	//srand(time(NULL));
	n = rand()%7;
        //scanf("%d", &n);

        long int num = 1;
        for (int i = 1; i <= n; i++){
                srand(time(NULL));
                num *= 10;
                num += rand()%(10);
        }

        while (!isPrime(num)){
                num ++;
        }

        return num;

}

//функция, которая ищет НОД двух чисел
long int NOD(long int a, long int b){
        while(a && b){
                if (a >= b)
                        a %= b;
                else
                        b %= a;
        }

        return (a | b);
}

//функция, которая ищет рандомное число из промежутка
// (1, fi-1), которое является взаимнопростым с fi
long int lookingForE(long int fi){

        srand(time(NULL));
        long int e = rand()%fi;

        long int a = fi;
        long int b = e;
        while ((NOD(a, b) != 1) && (!isPrime(e))){

                e++;
                a = fi;
                b = e;

        }
        return e;

}

//функция, которая ищет обратный элемент в кольце остатков fi
long int inverseE(long int fi, long int e){
        long int r[3];
        long int q;
        long int v[3];

        r[0] = fi;
        r[1] = e;

        v[0] = 0;
        v[1] = 1;

        while (r[2] != 1){
                q = r[0] / r[1];
                r[2] = r[0] % r[1];
                v[2] = v[0]  - q*v[1];
                v[0] = v[1];
                v[1] = v[2];
                r[0] = r[1];
                r[1] = r[2];
        }

        while (v[2] < 0){
                v[2] += fi;
        }

        return v[2];
}

//перевод из десятичной системы изсчисления в двоичную
long int ten2two(long int num){
        long int bin = 0;
        int k = 1;

        while (num) {
                bin += (num % 2) * k;
                k *= 10;
                num /= 2;
        }

        return bin;
}

//возведение a в степень b по модулю m
long int inPow(long int a, long int b, long int m){

        long int b_bin = ten2two(b);

	long int k = 1;
	while (b_bin / k){
		k *= 10;
	}

	long int p = a;
	p *= p;
	p -= (p/m)*m;
	k /= 10;

	while (k >= 100) {
		int b_i = (b_bin % k) / (k/10);
		if (b_i)
			p *= a;
		
		p *= p;
		p -= (p/m)*m;
		k /= 10;
	}

	int b_i = b_bin % 10;
	if (b_i)
		p *= a;

	p -= (p/m)*m;

	return p;
}

int main(){

        //герерируем рандомное простое число
        long int p = lookingForSimple();
        printf("p = %ld \n", p);

        //генерируем второе рандомное простое число
        long int q = lookingForSimple();
        printf("q = %ld \n", q);

        //вычисляем произведение простых чисел
        long int n = p*q;
        printf("n = %ld \n", n);

        //вычисляем функцию лапласа
        long int fi = (p-1)*(q-1);
        printf("fi = %ld \n", fi);

        //генерируем число е - взаимнопростое с fi из отрезка (1, fi-1)
        long int e = lookingForE(fi);
        printf("e = %ld \n", e);

        //ищем обратный элемент к е
        long int d = inverseE(fi, e);
        printf("e^(-1) = d = %ld \n", d);

        printf("%ld \n", e*d - ((e*d)/fi)*fi);

	//m - сообщение
	long int m = rand() % 100;
	printf("m = %ld \n", m);

	//зашифрованное m
	long int c = inPow(m, e, n);
	printf("c = %ld \n", c);

	//расшифровываем полученное сообщение с
	printf("m = %ld \n", inPow(c, d, n));

        return 0;

}

