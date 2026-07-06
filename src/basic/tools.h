int binary_search(int array[], int size, int target){

    int left = 0;
    int right = size - 1;
    while (left <= right){
        int middle = left+((right-left)/2);
        if (array[middle]>target){
            right = middle - 1;
        }
        else if (array[middle]<target){
            left = middle+1;
        }
        else {
            return middle;
        }
    }

    return -1;

}

int binary_search(vector<long int> array, int size, int target){

    int left = 0;
    int right = size - 1;
    while (left <= right){
        int middle = left+((right-left)/2);
        if (array[middle]>target){
            right = middle - 1;
        }
        else if (array[middle]<target){
            left = middle+1;
        }
        else {
            return middle;
        }
    }

    return -1;

}
