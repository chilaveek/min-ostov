using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace GraphOptimizationAnalysis
{
    // ═══════════════════════════════════════════════════════════════════════════
    // СТРУКТУРЫ ДАННЫХ ДЛЯ ГРАФА И РЁБЕР
    // ═══════════════════════════════════════════════════════════════════════════

    public class Edge : IComparable<Edge>
    {
        public int From { get; set; }
        public int To { get; set; }
        public int Weight { get; set; }

        public Edge(int from, int to, int weight)
        {
            From = from;
            To = to;
            Weight = weight;
        }

        public int CompareTo(Edge other)
        {
            return Weight.CompareTo(other.Weight);
        }
    }

    // Union-Find структура данных для алгоритма Крускала
    public class UnionFind
    {
        private int[] parent;
        private int[] rank;

        public UnionFind(int n)
        {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++)
            {
                parent[i] = i;
                rank[i] = 0;
            }
        }

        public int Find(int x)
        {
            if (parent[x] != x)
                parent[x] = Find(parent[x]); // Path compression
            return parent[x];
        }

        public bool Union(int x, int y)
        {
            int px = Find(x);
            int py = Find(y);

            if (px == py)
                return false;

            // Union by rank
            if (rank[px] < rank[py])
            {
                parent[px] = py;
            }
            else if (rank[px] > rank[py])
            {
                parent[py] = px;
            }
            else
            {
                parent[py] = px;
                rank[px]++;
            }

            return true;
        }
    }

    // Результаты исследования
    public class AlgorithmResult
    {
        public int GraphSize { get; set; }
        public long OperationCountKruskal { get; set; }
        public long OperationCountPrim { get; set; }
        public long TimeMillisecondsKruskal { get; set; }
        public long TimeMillisecondsPrim { get; set; }
        public double TheoreticalComplexityKruskal { get; set; }
        public double TheoreticalComplexityPrim { get; set; }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // АЛГОРИТМ КРУСКАЛА
    // ═══════════════════════════════════════════════════════════════════════════

    public class KruskalMST
    {
        public long OperationCount { get; private set; }

        public int FindMST(List<Edge> edges, int vertexCount)
        {
            OperationCount = 0;

            // Сортировка рёбер по весу: O(E log E)
            edges.Sort();
            OperationCount += (long)(edges.Count * Math.Log(edges.Count));

            UnionFind uf = new UnionFind(vertexCount);
            int mstWeight = 0;
            int edgesAdded = 0;

            foreach (Edge edge in edges)
            {
                OperationCount++; // ← СЧЁТЧИК ОПЕРАЦИЙ ВО ВНУТРЕННЕМ ЦИКЛЕ

                if (uf.Union(edge.From, edge.To))
                {
                    mstWeight += edge.Weight;
                    edgesAdded++;

                    if (edgesAdded == vertexCount - 1)
                        break;
                }
            }

            return mstWeight;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // АЛГОРИТМ ПРИМА
    // ═══════════════════════════════════════════════════════════════════════════

    public class PrimMST
    {
        public long OperationCount { get; private set; }

        public int FindMST(List<Edge> edges, int vertexCount)
        {
            OperationCount = 0;

            // Построение матрицы смежности
            int[,] adjacencyMatrix = new int[vertexCount, vertexCount];
            for (int i = 0; i < vertexCount; i++)
            {
                for (int j = 0; j < vertexCount; j++)
                {
                    adjacencyMatrix[i, j] = int.MaxValue;
                }
            }

            foreach (Edge edge in edges)
            {
                adjacencyMatrix[edge.From, edge.To] =
                    Math.Min(adjacencyMatrix[edge.From, edge.To], edge.Weight);
                adjacencyMatrix[edge.To, edge.From] =
                    Math.Min(adjacencyMatrix[edge.To, edge.From], edge.Weight);
            }

            // Алгоритм Прима: O(V²)
            bool[] inMST = new bool[vertexCount];
            int[] minCost = new int[vertexCount];

            for (int i = 0; i < vertexCount; i++)
                minCost[i] = int.MaxValue;

            minCost[0] = 0;
            int mstWeight = 0;

            for (int i = 0; i < vertexCount; i++)
            {
                // Найти вершину с минимальным весом
                int u = -1;
                int minWeight = int.MaxValue;

                for (int v = 0; v < vertexCount; v++)
                {
                    OperationCount++; // ← СЧЁТЧИК ОПЕРАЦИЙ ВО ВНУТРЕННЕМ ЦИКЛЕ

                    if (!inMST[v] && minCost[v] < minWeight)
                    {
                        minWeight = minCost[v];
                        u = v;
                    }
                }

                if (u == -1 || minWeight == int.MaxValue)
                    break;

                inMST[u] = true;
                mstWeight += minWeight;

                // Обновить стоимость соседних вершин
                for (int v = 0; v < vertexCount; v++)
                {
                    OperationCount++; // ← СЧЁТЧИК ОПЕРАЦИЙ

                    if (!inMST[v] && adjacencyMatrix[u, v] != int.MaxValue
                        && adjacencyMatrix[u, v] < minCost[v])
                    {
                        minCost[v] = adjacencyMatrix[u, v];
                    }
                }
            }

            return mstWeight;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ГЕНЕРАТОР ГРАФОВ
    // ═══════════════════════════════════════════════════════════════════════════

    public class GraphGenerator
    {
        private Random random = new Random(42);

        public List<Edge> GenerateConnectedGraph(int vertexCount, int maxWeight = 100)
        {
            List<Edge> edges = new List<Edge>();

            // Создание остовного дерева
            for (int i = 1; i < vertexCount; i++)
            {
                int from = random.Next(i);
                int weight = random.Next(1, maxWeight + 1);
                edges.Add(new Edge(from, i, weight));
            }

            // Добавление дополнительных рёбер
            int additionalEdges = Math.Max(vertexCount / 2, 10);
            for (int i = 0; i < additionalEdges; i++)
            {
                int from = random.Next(vertexCount);
                int to = random.Next(vertexCount);

                if (from != to)
                {
                    int weight = random.Next(1, maxWeight + 1);
                    edges.Add(new Edge(from, to, weight));
                }
            }

            return edges;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // АНАЛИЗАТОР ПРОИЗВОДИТЕЛЬНОСТИ
    // ═══════════════════════════════════════════════════════════════════════════

    public class PerformanceAnalyzer
    {
        private GraphGenerator generator = new GraphGenerator();

        public List<AlgorithmResult> AnalyzePerformance(int minSize, int maxSize, int step)
        {
            List<AlgorithmResult> results = new List<AlgorithmResult>();

            Console.WriteLine("╔════════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  ИССЛЕДОВАНИЕ ВЫЧИСЛИТЕЛЬНОЙ ЭФФЕКТИВНОСТИ АЛГОРИТМОВ МОП        ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════════════╝\n");

            for (int size = minSize; size <= maxSize; size += step)
            {
                Console.Write($"Анализ графа размером {size} вершин... ");

                // Генерируем граф
                List<Edge> edges = generator.GenerateConnectedGraph(size);

                // Алгоритм Крускала
                KruskalMST kruskal = new KruskalMST();
                Stopwatch swKruskal = Stopwatch.StartNew();
                kruskal.FindMST(new List<Edge>(edges), size);
                swKruskal.Stop();

                // Алгоритм Прима
                PrimMST prim = new PrimMST();
                Stopwatch swPrim = Stopwatch.StartNew();
                prim.FindMST(new List<Edge>(edges), size);
                swPrim.Stop();

                // Теоретическая сложность
                double theoreticalKruskal = size * Math.Log(size);
                double theoreticalPrim = size * size;

                AlgorithmResult result = new AlgorithmResult
                {
                    GraphSize = size,
                    OperationCountKruskal = kruskal.OperationCount,
                    OperationCountPrim = prim.OperationCount,
                    TimeMillisecondsKruskal = swKruskal.ElapsedMilliseconds,
                    TimeMillisecondsPrim = swPrim.ElapsedMilliseconds,
                    TheoreticalComplexityKruskal = theoreticalKruskal,
                    TheoreticalComplexityPrim = theoreticalPrim
                };

                results.Add(result);

                Console.WriteLine("✓");
                Console.WriteLine($"  Крускал: {kruskal.OperationCount:N0} операций");
                Console.WriteLine($"  Прим:    {prim.OperationCount:N0} операций\n");
            }

            return results;
        }

        public void PrintResults(List<AlgorithmResult> results)
        {
            Console.WriteLine("\n╔════════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ                                        ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════════════╝\n");

            Console.WriteLine("Размер | Крускал (опер) | Крускал (теор) | Прим (опер) | Прим (теор) | Соотнош.");
            Console.WriteLine("-------|----------------|----------------|-------------|-------------|----------");

            foreach (var result in results)
            {
                double ratioKruskal = result.OperationCountKruskal / (result.TheoreticalComplexityKruskal + 1);

                Console.WriteLine($"{result.GraphSize,5} | " +
                    $"{result.OperationCountKruskal,14:N0} | " +
                    $"{result.TheoreticalComplexityKruskal,14:F0} | " +
                    $"{result.OperationCountPrim,11:N0} | " +
                    $"{result.TheoreticalComplexityPrim,11:F0} | " +
                    $"{ratioKruskal:F2}");
            }

            Console.WriteLine();
        }

        public void PlotResults(List<AlgorithmResult> results)
        {
            Console.WriteLine("╔════════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  ГРАФИЧЕСКОЕ ПРЕДСТАВЛЕНИЕ: КРУСКАЛ                               ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════════════╝\n");

            PlotAlgorithm(results,
                r => r.OperationCountKruskal,
                r => r.TheoreticalComplexityKruskal,
                "Крускал (E log E)");

            Console.WriteLine("\n╔════════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  ГРАФИЧЕСКОЕ ПРЕДСТАВЛЕНИЕ: ПРИМ                                  ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════════════╝\n");

            PlotAlgorithm(results,
                r => r.OperationCountPrim,
                r => r.TheoreticalComplexityPrim,
                "Прим (V²)");
        }

        private void PlotAlgorithm(List<AlgorithmResult> results,
            Func<AlgorithmResult, long> experimentalFunc,
            Func<AlgorithmResult, double> theoreticalFunc,
            string title)
        {
            if (results.Count == 0)
                return;

            long maxExperimental = results.Max(r => experimentalFunc(r));
            double maxTheoretical = results.Max(r => theoreticalFunc(r));
            double maxValue = Math.Max(maxExperimental, (long)maxTheoretical);

            const int chartWidth = 60;

            Console.WriteLine($"Легенда: █ Экспериментальные данные, ▓ Теоретическая оценка");
            Console.WriteLine(new string('─', chartWidth + 10));

            foreach (var result in results)
            {
                long experimental = experimentalFunc(result);
                double theoretical = theoreticalFunc(result);

                int expBarLength = (int)((experimental / (double)maxValue) * chartWidth);
                int theoBarLength = (int)((theoretical / maxValue) * chartWidth);

                string expBar = new string('█', expBarLength);
                string theoBar = new string('▓', theoBarLength);

                Console.Write($"V={result.GraphSize,4} | ");
                Console.Write(expBar);
                Console.Write(new string(' ', Math.Max(0, chartWidth - expBarLength)));
                Console.Write($" | {experimental,10:N0}\n");

                Console.Write($"     | ");
                Console.Write(theoBar);
                Console.Write(new string(' ', Math.Max(0, chartWidth - theoBarLength)));
                Console.WriteLine($" | {theoretical,10:F0}");
                Console.WriteLine();
            }

            Console.WriteLine(new string('─', chartWidth + 10));
            Console.WriteLine("\n📊 АНАЛИЗ СЛОЖНОСТИ:\n");

            if (results.Count >= 2)
            {
                var first = results.First();
                var last = results.Last();

                double experimentalGrowth = (double)experimentalFunc(last) / experimentalFunc(first);
                double sizeGrowth = (double)last.GraphSize / first.GraphSize;

                Console.WriteLine($"  Прирост размера: {sizeGrowth:F2}x");
                Console.WriteLine($"  Экспериментальный прирост операций: {experimentalGrowth:F2}x");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ГЛАВНАЯ ПРОГРАММА
    // ═══════════════════════════════════════════════════════════════════════════

    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            // ПАРАМЕТРЫ ИССЛЕДОВАНИЯ
            int minGraphSize = 10;    // Нижняя граница размерности
            int maxGraphSize = 100;   // Верхняя граница размерности
            int step = 10;            // Шаг итерации

            PerformanceAnalyzer analyzer = new PerformanceAnalyzer();

            // Цикл исполнения алгоритма в диапазоне размерностей
            List<AlgorithmResult> results = analyzer.AnalyzePerformance(minGraphSize, maxGraphSize, step);

            // Вывод и визуализация результатов
            analyzer.PrintResults(results);
            analyzer.PlotResults(results);

            Console.WriteLine("\nИсследование завершено!");
            Console.WriteLine("Нажмите Enter для выхода...");
            Console.ReadLine();
        }
    }
}
