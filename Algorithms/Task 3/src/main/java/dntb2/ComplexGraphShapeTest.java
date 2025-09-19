package dntb2;

import java.io.*;

public class ComplexGraphShapeTest {

    public static void main(String[] args) {
        testGraphWithRedundantEdges();
        testLargeComplexGraph();
        testInterleavedChains();
        testComplexCycleDetection();
        testGraphWithParallelPaths();
        testGraphWithConvergenceAndDivergence();
        testGraphWithDisconnectedComponents();
        testGraphWithNestedDiamondStructures();
        testGraphWithLongChain();
        testGraphWithMultipleSourcesSinks();
    }

    // Utility: Write the given content to a file.
    private static void writeFile(String filename, String content) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write(content);
        } catch (IOException e) {
            System.out.println("Error writing file " + filename + ": " + e.getMessage());
        }
    }

    // Utility: Run the solver on the given file and return its output as a String.
    private static String runSolver(String filename) {
        Solver solver = new Solver();
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(baos));
        try {
            solver.solve(filename);
        } catch (IOException e) {
            System.setOut(originalOut);
            return "IOException: " + e.getMessage();
        }
        System.out.flush();
        System.setOut(originalOut);
        return baos.toString();
    }

    /**
     * Test a large, layered complex graph.
     * Graph (10 tasks):
     *   1 -> 2, 1 -> 3
     *   2 -> 4, 3 -> 4
     *   4 -> 5
     *   2 -> 6, 6 -> 7
     *   4 -> 8, 7 -> 8
     *   8 -> 9, 5 -> 9
     *   9 -> 10
     * Expected longest chain: 1,2,6,7,8,9,10 (7 tasks)
     */
    private static void testLargeComplexGraph() {
        String filename = "complex_large.txt";
        String content = "10\n" +
                "1,2\n" +
                "1,3\n" +
                "2,4\n" +
                "3,4\n" +
                "4,5\n" +
                "2,6\n" +
                "6,7\n" +
                "4,8\n" +
                "7,8\n" +
                "8,9\n" +
                "5,9\n" +
                "9,10\n";
        writeFile(filename, content);
        String output = runSolver(filename);
        System.out.println("----- Test Large Complex Graph Output -----");
        System.out.println(output);
        if (output.contains("THE LONGEST DEPENDENCY CHAIN IS 7 TASKS")) {
            System.out.println("Test Large Complex Graph: PASS");
        } else {
            System.out.println("Test Large Complex Graph: FAIL");
        }
    }

    /**
     * Test interleaved chains.
     * Graph (8 tasks):
     *   1 -> 3, 2 -> 3,
     *   3 -> 4, 3 -> 5,
     *   4 -> 6, 5 -> 6,
     *   6 -> 7, 6 -> 8,
     *   7 -> 8.
     * Expected longest chain: e.g. 1,3,4,6,7,8 (6 tasks)
     */
    private static void testInterleavedChains() {
        String filename = "complex_interleaved.txt";
        String content = "8\n" +
                "1,3\n" +
                "2,3\n" +
                "3,4\n" +
                "3,5\n" +
                "4,6\n" +
                "5,6\n" +
                "6,7\n" +
                "6,8\n" +
                "7,8\n";
        writeFile(filename, content);
        String output = runSolver(filename);
        System.out.println("----- Test Interleaved Chains Graph Output -----");
        System.out.println(output);
        if (output.contains("THE LONGEST DEPENDENCY CHAIN IS 6 TASKS")) {
            System.out.println("Test Interleaved Chains: PASS");
        } else {
            System.out.println("Test Interleaved Chains: FAIL");
        }
    }

    /**
     * Test a graph with an embedded cycle.
     * Graph (6 tasks):
     *   1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5,
     *   5 -> 3 (cycle among 3-4-5), and 2 -> 6.
     * Expected: IMPOSSIBLE
     */
    private static void testComplexCycleDetection() {
        String filename = "complex_cycle.txt";
        String content = "6\n" +
                "1,2\n" +
                "2,3\n" +
                "3,4\n" +
                "4,5\n" +
                "5,3\n" + // creates cycle
                "2,6\n";
        writeFile(filename, content);
        String output = runSolver(filename).trim();
        System.out.println("----- Test Complex Cycle Detection Output -----");
        System.out.println(output);
        if (output.equals("IMPOSSIBLE")) {
            System.out.println("Test Complex Cycle Detection: PASS");
        } else {
            System.out.println("Test Complex Cycle Detection: FAIL");
        }
    }

    /**
     * Test a graph with many redundant edges.
     * Graph (5 tasks):
     *   Multiple redundant edges between tasks.
     * Expected longest chain: 1,2,3,4,5 (5 tasks)
     */
    private static void testGraphWithRedundantEdges() {
        String filename = "complex_redundant.txt";
        String content = "5\n" +
                "1,2\n" +
                "1,2\n" +
                "1,3\n" +
                "2,3\n" +
                "2,3\n" +
                "3,4\n" +
                "4,5\n" +
                "1,4\n" +
                "2,4\n";
        writeFile(filename, content);
        String output = runSolver(filename);
        System.out.println("----- Test Graph With Redundant Edges Output -----");
        System.out.println(output);
        if (output.contains("THE LONGEST DEPENDENCY CHAIN IS 5 TASKS")) {
            System.out.println("Test Graph With Redundant Edges: PASS");
        } else {
            System.out.println("Test Graph With Redundant Edges: FAIL");
        }
    }

    /**
     * Test a graph with multiple parallel paths.
     * Graph (9 tasks):
     *   1 -> 2, 1 -> 3,
     *   2 -> 4, 3 -> 4,
     *   4 -> 5, 4 -> 6,
     *   5 -> 7, 6 -> 7,
     *   7 -> 8, 7 -> 9.
     * Expected longest chain: e.g., 1,2,4,5,7,8 (6 tasks)
     */
    private static void testGraphWithParallelPaths() {
        String filename = "complex_parallel.txt";
        String content = "9\n" +
                "1,2\n" +
                "1,3\n" +
                "2,4\n" +
                "3,4\n" +
                "4,5\n" +
                "4,6\n" +
                "5,7\n" +
                "6,7\n" +
                "7,8\n" +
                "7,9\n";
        writeFile(filename, content);
        String output = runSolver(filename);
        System.out.println("----- Test Graph With Parallel Paths Output -----");
        System.out.println(output);
        if (output.contains("THE LONGEST DEPENDENCY CHAIN IS 6 TASKS")) {
            System.out.println("Test Graph With Parallel Paths: PASS");
        } else {
            System.out.println("Test Graph With Parallel Paths: FAIL");
        }
    }

    /**
     * Test a graph with both convergence and divergence.
     * Graph (8 tasks):
     *   1 -> 4, 2 -> 4, 3 -> 4,
     *   4 -> 5, 4 -> 6,
     *   5 -> 7, 6 -> 7,
     *   7 -> 8, and extra edge 4 -> 8.
     * Expected longest chain: e.g., 1,4,5,7,8 (5 tasks)
     */
    private static void testGraphWithConvergenceAndDivergence() {
        String filename = "complex_conv_div.txt";
        String content = "8\n" +
                "1,4\n" +
                "2,4\n" +
                "3,4\n" +
                "4,5\n" +
                "4,6\n" +
                "5,7\n" +
                "6,7\n" +
                "7,8\n" +
                "4,8\n";
        writeFile(filename, content);
        String output = runSolver(filename);
        System.out.println("----- Test Graph With Convergence and Divergence Output -----");
        System.out.println(output);
        if (output.contains("THE LONGEST DEPENDENCY CHAIN IS 5 TASKS")) {
            System.out.println("Test Graph With Convergence and Divergence: PASS");
        } else {
            System.out.println("Test Graph With Convergence and Divergence: FAIL");
        }
    }

    /**
     * Test a graph with disconnected components.
     * Graph (6 tasks):
     *   Component 1: 1 -> 2 -> 3,
     *   Component 2: 4 -> 5,
     *   Task 6 is isolated.
     * Expected longest chain: 3 tasks.
     */
    private static void testGraphWithDisconnectedComponents() {
        String filename = "complex_disconnected.txt";
        String content = "6\n" +
                "1,2\n" +
                "2,3\n" +
                "4,5\n";
        writeFile(filename, content);
        String output = runSolver(filename);
        System.out.println("----- Test Graph With Disconnected Components Output -----");
        System.out.println(output);
        if (output.contains("THE LONGEST DEPENDENCY CHAIN IS 3 TASKS")) {
            System.out.println("Test Graph With Disconnected Components: PASS");
        } else {
            System.out.println("Test Graph With Disconnected Components: FAIL");
        }
    }

    /**
     * Test a graph with nested diamond structures.
     * Graph (12 tasks):
     *   Diamond 1: 1 -> 2, 1 -> 3; 2 -> 4, 3 -> 4.
     *   Diamond 2: 4 -> 5, 4 -> 6; 5 -> 7, 6 -> 7.
     *   Diamond 3: 7 -> 8, 7 -> 9; 8 -> 10, 9 -> 10.
     * Expected longest chain: 1,2,4,5,7,8,10 (7 tasks)
     */
    private static void testGraphWithNestedDiamondStructures() {
        String filename = "complex_nested_diamond.txt";
        String content = "12\n" +
                "1,2\n" +
                "1,3\n" +
                "2,4\n" +
                "3,4\n" +
                "4,5\n" +
                "4,6\n" +
                "5,7\n" +
                "6,7\n" +
                "7,8\n" +
                "7,9\n" +
                "8,10\n" +
                "9,10\n" +
                "10,11\n" +
                "10,12\n";
        // In this nested structure, one possible longest chain is:
        // 1,2,4,5,7,8,10,11 (8 tasks) or 1,2,4,5,7,8,10,12 (8 tasks)
        writeFile(filename, content);
        String output = runSolver(filename);
        System.out.println("----- Test Graph With Nested Diamond Structures Output -----");
        System.out.println(output);
        if (output.contains("THE LONGEST DEPENDENCY CHAIN IS 8 TASKS")) {
            System.out.println("Test Graph With Nested Diamond Structures: PASS");
        } else {
            System.out.println("Test Graph With Nested Diamond Structures: FAIL");
        }
    }

    /**
     * Test a long linear chain with extra edges.
     * Graph (12 tasks):
     *   Chain: 1->2->3->...->12, plus extra edges 1->5 and 3->7.
     * Expected longest chain: 12 tasks.
     */
    private static void testGraphWithLongChain() {
        String filename = "complex_long_chain.txt";
        StringBuilder sb = new StringBuilder();
        int n = 12;
        sb.append(n).append("\n");
        for (int i = 1; i < n; i++) {
            sb.append(i).append(",").append(i + 1).append("\n");
        }
        sb.append("1,5\n");
        sb.append("3,7\n");
        String content = sb.toString();
        writeFile(filename, content);
        String output = runSolver(filename);
        System.out.println("----- Test Graph With Long Chain Output -----");
        System.out.println(output);
        if (output.contains("THE LONGEST DEPENDENCY CHAIN IS 12 TASKS")) {
            System.out.println("Test Graph With Long Chain: PASS");
        } else {
            System.out.println("Test Graph With Long Chain: FAIL");
        }
    }

    /**
     * Test a graph with multiple sources and sinks.
     * Graph (9 tasks):
     *   Sources: 1, 2, 3 (independent)
     *   Merge: 1 -> 4, 2 -> 4; then 4 -> 5, and 3 -> 5;
     *   Divergence: 5 -> 6, 5 -> 7;
     *   Convergence: 6 -> 8, 7 -> 8; 8 -> 9.
     * Expected longest chain: e.g., 1,4,5,6,8,9 (6 tasks)
     */
    private static void testGraphWithMultipleSourcesSinks() {
        String filename = "complex_multi_source_sink.txt";
        String content = "9\n" +
                "1,4\n" +
                "2,4\n" +
                "3,5\n" +
                "4,5\n" +
                "5,6\n" +
                "5,7\n" +
                "6,8\n" +
                "7,8\n" +
                "8,9\n";
        writeFile(filename, content);
        String output = runSolver(filename);
        System.out.println("----- Test Graph With Multiple Sources and Sinks Output -----");
        System.out.println(output);
        if (output.contains("THE LONGEST DEPENDENCY CHAIN IS 6 TASKS")) {
            System.out.println("Test Graph With Multiple Sources and Sinks: PASS");
        } else {
            System.out.println("Test Graph With Multiple Sources and Sinks: FAIL");
        }
    }
}
