package dntb2;

public class HeapBased implements Algs202425Tick1{

    private static class MinHeap {
        ListCell[] arr;
        int size;

        MinHeap(ListCell[] arr) {
            this.arr = arr;
            this.size = arr.length;
            fullHeapify();
        }

        private void swap(int i, int j) {
            ListCell temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }

        private void reheapify(int i) {
            int l = 2*i;
            int r = l + 1;

            int min = i;
            if (l < size && arr[l].val < arr[i].val) {
                min = l;
            }
            if (r < size && arr[r].val < arr[min].val) {
                min = r;
            }

            if (min != i) {
                swap(i, min);
                reheapify(min);
            }
        }

        private void fullHeapify() {
            int start = size / 2;

            for (int i = start; i >= 0; i--) {
                reheapify(i);
            }
        }

        public ListCell extractMin() {
            if (size == 0) {
                return null;
            }

            ListCell min = arr[0];
            swap(0, size-1);
            size -= 1;
            reheapify(0);
            return min;
        }

        public ListCell replaceMinNext() {
            if (size == 0) {
                return null;
            }

            ListCell temp = arr[0];
            if (temp.nxt == null) {
                return extractMin();
            }
            else {
                arr[0] = temp.nxt;
                reheapify(0);
                return temp;
            }
        }

    }

    public static void printList(ListCell head) {
        ListCell current = head;
        StringBuilder sb = new StringBuilder();
        while (current != null) {
            sb.append(current.val).append(" -> ");
            current = current.nxt;
        }
        sb.append("null");
        System.out.println(sb.toString());
    }

    /*
    Split into r sorted linked lists
    Add first element from each linked list into array
    Heapify array
    Extract min and put at end of linked list
    Do this n times
    Reverse list
     */

    @Override
    public ListCell sort(ListCell head) {
        if (head == null) {
            return null;
        }

        int splits = 1;
        ListCell current = head;

        while (current.nxt != null) {
            if (current.nxt.val < current.val) {
                splits += 1;
            }
            current = current.nxt;
        }

        ListCell[] heads = new ListCell[splits];

        int split = 0;
        current = head;
        heads[0] = current;


        while (current.nxt != null) {
            if (current.nxt.val < current.val) {
                split += 1;
                heads[split] = current.nxt;
                current.nxt = null;
                current = heads[split];
            }
            else {
                current = current.nxt;
            }
        }

        if (splits == 1) {
            return heads[0];
        }

        MinHeap minHeap = new MinHeap(heads);

        head = new ListCell();
        ListCell tail = head;

        ListCell min = minHeap.replaceMinNext();
        while (min != null) {
            min.nxt = null;
            tail.nxt = min;
            tail = tail.nxt;
            min = minHeap.replaceMinNext();
        }

        return head.nxt;
    }
}


