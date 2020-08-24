int f(int n) {
    char exampleArray[] = { 'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd' };
    for (int i = 0; i <= 11; i++) {
        System.out.println(exampleArray[i]);
    }
    if (n == 0) {
        return 1; 
    } else {
        return n * f(n-1);
    }
}