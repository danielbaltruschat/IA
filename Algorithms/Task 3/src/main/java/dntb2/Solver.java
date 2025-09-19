package dntb2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class Solver implements Algs202425Tick3{
    private static class LinkedList<T> {
        public LinkedList<T> next;
        public T value;
    }

    private static class Queue {
        private final int[] arr;
        private int front = 0;
        private int rear = 0;

        public Queue(int size) {
            arr = new int[size + 1];
        }

        public boolean isEmpty() {
            return front == rear;
        }

        public boolean isFull() {
            return (rear + 1) % arr.length == front;
        }


        public void enqueue(int item) {
            if (!isFull()) {
                arr[rear] = item;
                rear = (rear + 1) % arr.length;
            }
            else {
                throw new NoSuchElementException();
            }
        }


        public int dequeue() {
            if (!isEmpty()) {
                int item = arr[front];
                front = (front + 1) % arr.length;
                return item;
            }
            else {
                throw new NoSuchElementException();
            }
        }

    }

    //n1 -> n2
    private int insertIntoAdjList(LinkedList<Integer>[] adjList, int n1, int n2) {
        if (adjList[n1] == null) {
            adjList[n1] = new LinkedList<Integer>();
            adjList[n1].value = n2;
            return 0;
        }
        else {
            LinkedList<Integer> current = adjList[n1];

            while (current.next != null) {
                if (current.value == n2) {
                    return 1;
                }
                current = current.next;
            }

            if (current.value == n2) return 1;

            current.next = new LinkedList<Integer>();
            current.next.value = n2;
            return 0;
        }
    }

    private String LLToString(LinkedList<Integer> l) {
        String s = "";
        while (l != null) {
            s += Integer.toString(l.value + 1);
            s += "\n";
            l = l.next;
        }
        return s;
    }

    private int max(int[] arr) {
        int max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) max = arr[i];
        }
        return max;
    }

    //Use previously computed values
    private int DFS(int node, LinkedList<Integer>[] adjList, int[] chainLengths) {
        if (chainLengths[node] != 0) {
            return chainLengths[node];
        }

        if (adjList[node] == null) {
            chainLengths[node] = 1;
            return 1;
        }

        int max = 0;
        LinkedList<Integer> nexts = adjList[node];
        while (nexts != null) {
            int len = DFS(nexts.value, adjList, chainLengths);
            if (len > max) {
                max = len;
            }
            nexts = nexts.next;
        }

        chainLengths[node] = max + 1;
        return max + 1;
    }


    public void solve(String filename) throws IOException {
        File file;
        Scanner reader;
        try {
            file = new File(filename);
            reader = new Scanner(file);
        }
        catch (FileNotFoundException e) {
            throw new IOException(e);
        }

        int n = Integer.parseInt(reader.nextLine());


        LinkedList<Integer>[] adjList = new LinkedList[n];
        LinkedList<Integer>[] backwards = new LinkedList[n];

        int[] prereqs = new int[n];


        while (reader.hasNextLine()) {
            String line = reader.nextLine();
            String[] numbers = line.split(",");

            int n1 = Integer.parseInt(numbers[0]) - 1;

            int n2 = Integer.parseInt(numbers[1]) - 1;

            insertIntoAdjList(adjList, n1, n2);
            int flag = insertIntoAdjList(backwards, n2, n1);
            if (flag == 0) prereqs[n2] += 1;
        }


        LinkedList<Integer> order = null;
        LinkedList<Integer> orderLast = order;
        int[] chainLengths = new int[n];

        for (int i = 0; i < n; i++) {
            if (backwards[i] == null) {
                Queue q = new Queue(n);
                q.enqueue(i);

                while (!q.isEmpty()) {
                    int current  = q.dequeue();

                    if (order == null) {
                        order = new LinkedList<Integer>();
                        order.value = current;
                        orderLast = order;
                    }
                    else {
                        orderLast.next = new LinkedList<Integer>();
                        orderLast = orderLast.next;
                        orderLast.value = current;
                    }

                    LinkedList<Integer> nexts = adjList[current];

                    // traverse LL
                    int val;
                    while (nexts != null) {
                        val = nexts.value;

                        prereqs[val] -= 1;
                        if (prereqs[val] == 0) {
                            q.enqueue(val);
                        }

                        nexts = nexts.next;

                    }
                }

            }
        }


        //check for cycles
        for (int i = 0; i < n; i++) {
            if (prereqs[i] != 0) {
                System.out.println("IMPOSSIBLE");
                return;
            }
        }

        //find longest chain
        for (int i = 0; i < n; i++) {
            if (backwards[i] == null) {
                DFS(i, adjList, chainLengths);
            }
        }


        System.out.println(LLToString(order));


        System.out.println("THE LONGEST DEPENDENCY CHAIN IS " + max(chainLengths) + " TASKS");
    }

}
