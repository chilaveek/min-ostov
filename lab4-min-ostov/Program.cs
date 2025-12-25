using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace MSTAlgorithmsResearch
{
    public class Edge : IComparable<Edge>
    {
        public int U { get; set; }
        public int V { get; set; }
        public double Weight { get; set; }

        public Edge(int u, int v, double weight)
        {
            U = u;
            V = v;
            Weight = weight;
        }

        public int CompareTo(Edge other)
        {
            return Weight.CompareTo(other.Weight);
        }

        public override string ToString()
        {
            return $"Edge({U}-{V}, w={Weight:F2})";
        }
    }
    public class UnionFind
    {
        private int[] parent;
        private int[] rank;
        public long Operations { get; set; }

        public UnionFind(int n)
        {
            parent = new int[n];
            rank = new int[n];
            Operations = 0;

            for (int i = 0; i < n; i++)
            {
                parent[i] = i;
                rank[i] = 0;
            }
        }

        /// <summary>
        /// Найти корневой элемент множества с путь сжатия
        /// </summary>
        public int Find(int x)
        {
            Operations++;
            if (parent[x] != x)
            {
                parent[x] = Find(parent[x]);  // Путь сжатия
            }
            return parent[x];
        }

        /// <summary>
        /// Объединить два множества. Возвращает true если они были разными
        /// </summary>
        public bool Union(int x, int y)
        {
            int rootX = Find(x);
            int rootY = Find(y);

            if (rootX == rootY)
                return false;

            Operations++;

            // Union by rank
            if (rank[rootX] < rank[rootY])
            {
                parent[rootX] = rootY;
            }
            else if (rank[rootX] > rank[rootY])
            {
                parent[rootY] = rootX;
            }
            else
            {
                parent[rootY] = rootX;
                rank[rootX]++;
            }

            return true;
        }
    }
    public class Graph
    {
        public int VertexCount { get; private set; }
        public List<Edge> Edges { get; private set; }
        public List<Tuple<int, double>>[] AdjacencyList { get; private set; }

        public Graph(int vertexCount)
        {
            VertexCount = vertexCount;
            Edges = new List<Edge>();
            AdjacencyList = new List<Tuple<int, double>>[vertexCount];

            for (int i = 0; i < vertexCount; i++)
            {
                AdjacencyList[i] = new List<Tuple<int, double>>();
            }
        }

        /// <summary>
        /// Добавить ребро в граф
        /// </summary>
        public void AddEdge(int u, int v, double weight)
        {
            Edges.Add(new Edge(u, v, weight));
            AdjacencyList[u].Add(new Tuple<int, double>(v, weight));
            AdjacencyList[v].Add(new Tuple<int, double>(u, weight));
        }

        /// <summary>
        /// Получить ребра, отсортированные по весу
        /// </summary>
        public List<Edge> GetEdgesSortedByWeight()
        {
            var sortedEdges = new List<Edge>(Edges);
            sortedEdges.Sort();
            return sortedEdges;
        }

        /// <summary>
        /// Вычислить вес минимального остова
        /// </summary>
        public double GetMSTWeight()
        {
            var edges = GetEdgesSortedByWeight();
            var uf = new UnionFind(VertexCount);
            double totalWeight = 0.0;
            int edgeCount = 0;

            foreach (var edge in edges)
            {
                if (uf.Union(edge.U, edge.V))
                {
                    totalWeight += edge.Weight;
                    edgeCount++;
                    if (edgeCount == VertexCount - 1)
                        break;
                }
            }

            return totalWeight;
        }
    }

    // ЧАСТЬ 2: ГЕНЕРАТОР ДАННЫХ
    public static class DataGenerator
    {
        private static Random random = new Random();

        /// <summary>
        /// Генерировать полный граф (complete graph)
        /// </summary>
        public static Graph GenerateCompleteGraph(int vertexCount, double maxWeight = 100.0, int? seed = null)
        {
            if (seed.HasValue)
                random = new Random(seed.Value);

            var graph = new Graph(vertexCount);

            // Добавляем все ребра между вершинами
            for (int i = 0; i < vertexCount; i++)
            {
                for (int j = i + 1; j < vertexCount; j++)
                {
                    double weight = random.NextDouble() * maxWeight + 1;
                    graph.AddEdge(i, j, weight);
                }
            }

            return graph;
        }

        /// <summary>
        /// Генерировать разреженный граф
        /// </summary>
        public static Graph GenerateSparseGraph(int vertexCount, double edgeProbability = 0.3,
                                               double maxWeight = 100.0, int? seed = null)
        {
            if (seed.HasValue)
                random = new Random(seed.Value);

            var graph = new Graph(vertexCount);

            // Сначала добавляем ребра для связности
            var vertices = Enumerable.Range(0, vertexCount).ToList();
            for (int i = 0; i < vertices.Count - 1; i++)
            {
                double weight = random.NextDouble() * maxWeight + 1;
                graph.AddEdge(vertices[i], vertices[i + 1], weight);
            }

            // Затем добавляем остальные ребра с вероятностью
            for (int i = 0; i < vertexCount; i++)
            {
                for (int j = i + 1; j < vertexCount; j++)
                {
                    if (random.NextDouble() < edgeProbability)
                    {
                        double weight = random.NextDouble() * maxWeight + 1;
                        graph.AddEdge(i, j, weight);
                    }
                }
            }

            return graph;
        }

        /// <summary>
        /// Генерировать граф с евклидовыми расстояниями
        /// </summary>
        public static Graph GenerateEuclideanGraph(int vertexCount, double maxCoord = 100.0, int? seed = null)
        {
            if (seed.HasValue)
                random = new Random(seed.Value);

            var graph = new Graph(vertexCount);

            // Генерируем случайные координаты для вершин
            var vertexCoords = new List<Tuple<double, double>>();
            for (int i = 0; i < vertexCount; i++)
            {
                vertexCoords.Add(new Tuple<double, double>(
                    random.NextDouble() * maxCoord,
                    random.NextDouble() * maxCoord
                ));
            }

            // Добавляем ребра между всеми парами с евклидовым расстоянием
            for (int i = 0; i < vertexCount; i++)
            {
                for (int j = i + 1; j < vertexCount; j++)
                {
                    double x1 = vertexCoords[i].Item1;
                    double y1 = vertexCoords[i].Item2;
                    double x2 = vertexCoords[j].Item1;
                    double y2 = vertexCoords[j].Item2;

                    double distance = Math.Sqrt(Math.Pow(x2 - x1, 2) + Math.Pow(y2 - y1, 2));
                    graph.AddEdge(i, j, distance);
                }
            }

            return graph;
        }
    }

    // ЧАСТЬ 3: АЛГОРИТМЫ МИНИМАЛЬНОГО ОСТОВА
    public class KruskalAlgorithm
    {
        public long Operations { get; private set; }

        /// <summary>
        /// Найти минимальный остов с помощью алгоритма Крускала
        /// </summary>
        public Tuple<List<Edge>, double> FindMST(Graph graph)
        {
            Operations = 0;

            // Сортируем ребра по весу
            var edges = graph.GetEdgesSortedByWeight();

            // Инициализируем Union-Find
            var uf = new UnionFind(graph.VertexCount);

            var mstEdges = new List<Edge>();
            double totalWeight = 0.0;

            // Основной цикл алгоритма
            foreach (var edge in edges)
            {
                Operations++;  // Счетчик операций во внутреннем цикле

                // Проверяем, создает ли ребро цикл
                if (uf.Union(edge.U, edge.V))
                {
                    mstEdges.Add(edge);
                    totalWeight += edge.Weight;

                    // Если добавили N-1 ребер, минимальный остов найден
                    if (mstEdges.Count == graph.VertexCount - 1)
                        break;
                }
            }

            // Добавляем операции из Union-Find
            Operations += uf.Operations;

            return new Tuple<List<Edge>, double>(mstEdges, totalWeight);
        }
    }
    public class PrimAlgorithm
    {
        public long Operations { get; private set; }

        /// <summary>
        /// Найти минимальный остов с помощью алгоритма Прима
        /// </summary>
        public Tuple<List<Edge>, double> FindMST(Graph graph)
        {
            Operations = 0;

            int n = graph.VertexCount;

            // Множество вершин, уже включенных в MST
            var inMST = new bool[n];

            // Минимальный ключ (вес) для каждой вершины
            var key = new double[n];
            for (int i = 0; i < n; i++)
                key[i] = double.MaxValue;

            // Родитель вершины в MST
            var parent = new int[n];
            for (int i = 0; i < n; i++)
                parent[i] = -1;

            // Стартуем с первой вершины
            key[0] = 0;

            var mstEdges = new List<Edge>();
            double totalWeight = 0.0;

            // Основной цикл: добавляем N-1 ребер
            for (int count = 0; count < n - 1; count++)
            {
                // Найти вершину с минимальным ключом, не включенную в MST
                double minKey = double.MaxValue;
                int minVertex = -1;

                for (int v = 0; v < n; v++)
                {
                    Operations++;  // Счетчик операций во внутреннем цикле

                    if (!inMST[v] && key[v] < minKey)
                    {
                        minKey = key[v];
                        minVertex = v;
                    }
                }

                // Добавляем вершину в MST
                inMST[minVertex] = true;

                // Если это не первая вершина, добавляем ребро
                if (parent[minVertex] != -1)
                {
                    var edge = new Edge(parent[minVertex], minVertex, key[minVertex]);
                    mstEdges.Add(edge);
                    totalWeight += key[minVertex];
                }

                // Обновляем ключи соседних вершин
                foreach (var neighbor in graph.AdjacencyList[minVertex])
                {
                    Operations++;  // Счетчик операций

                    int neighborVertex = neighbor.Item1;
                    double weight = neighbor.Item2;

                    if (!inMST[neighborVertex] && weight < key[neighborVertex])
                    {
                        key[neighborVertex] = weight;
                        parent[neighborVertex] = minVertex;
                    }
                }
            }

            return new Tuple<List<Edge>, double>(mstEdges, totalWeight);
        }
    }
    public class PrimAlgorithmHeap
    {
        public long Operations { get; private set; }

        private class HeapItem : IComparable<HeapItem>
        {
            public double Weight { get; set; }
            public int FromVertex { get; set; }
            public int ToVertex { get; set; }

            public int CompareTo(HeapItem other)
            {
                return Weight.CompareTo(other.Weight);
            }
        }

        /// <summary>
        /// Найти минимальный остов с помощью алгоритма Прима с кучей
        /// </summary>
        public Tuple<List<Edge>, double> FindMST(Graph graph)
        {
            Operations = 0;

            int n = graph.VertexCount;

            // Множество вершин, уже включенных в MST
            var inMST = new bool[n];

            // Приоритетная очередь (используем SortedSet как имитацию кучи)
            var pq = new SortedSet<HeapItem>(new HeapComparer());
            pq.Add(new HeapItem { Weight = 0, FromVertex = -1, ToVertex = 0 });

            var mstEdges = new List<Edge>();
            double totalWeight = 0.0;

            // Основной цикл
            while (pq.Count > 0)
            {
                var item = pq.Min;
                pq.Remove(item);

                Operations++;  // Счетчик операций во внутреннем цикле

                // Пропускаем, если вершина уже в MST
                if (inMST[item.ToVertex])
                    continue;

                // Добавляем вершину в MST
                inMST[item.ToVertex] = true;

                // Если это не начальная вершина, добавляем ребро
                if (item.FromVertex != -1)
                {
                    var edge = new Edge(item.FromVertex, item.ToVertex, item.Weight);
                    mstEdges.Add(edge);
                    totalWeight += item.Weight;
                }

                // Добавляем соседей в приоритетную очередь
                foreach (var neighbor in graph.AdjacencyList[item.ToVertex])
                {
                    Operations++;  // Счетчик операций

                    int neighborVertex = neighbor.Item1;
                    double edgeWeight = neighbor.Item2;

                    if (!inMST[neighborVertex])
                    {
                        pq.Add(new HeapItem
                        {
                            Weight = edgeWeight,
                            FromVertex = item.ToVertex,
                            ToVertex = neighborVertex
                        });
                    }
                }
            }

            return new Tuple<List<Edge>, double>(mstEdges, totalWeight);
        }

        private class HeapComparer : IComparer<HeapItem>
        {
            public int Compare(HeapItem x, HeapItem y)
            {
                int result = x.Weight.CompareTo(y.Weight);
                if (result != 0)
                    return result;

                result = x.ToVertex.CompareTo(y.ToVertex);
                if (result != 0)
                    return result;

                return x.FromVertex.CompareTo(y.FromVertex);
            }
        }
    }

    // ЧАСТЬ 4: АНАЛИЗ СЛОЖНОСТИ
    public static class ComplexityAnalyzer
    {
        /// <summary>
        /// Теоретическая сложность Крускала: O(E*log(E))
        /// </summary>
        public static double TheoreticalKruskal(int vertexCount, int edgeCount)
        {
            return edgeCount * Math.Log(edgeCount + 1);
        }

        /// <summary>
        /// Теоретическая сложность Прима (простая реализация): O(V^2)
        /// </summary>
        public static double TheoreticalPrimSimple(int vertexCount)
        {
            return vertexCount * vertexCount;
        }

        /// <summary>
        /// Теоретическая сложность Прима с кучей: O((V+E)*log(V))
        /// </summary>
        public static double TheoreticalPrimHeap(int vertexCount, int edgeCount)
        {
            return (vertexCount + edgeCount) * Math.Log(vertexCount + 1);
        }
    }

    // ЧАСТЬ 5: ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТОВ
    public class PerformanceExperiment
    {
        public Dictionary<string, List<int>> Dimensions { get; private set; }
        public Dictionary<string, List<long>> Operations { get; private set; }

        public PerformanceExperiment()
        {
            Dimensions = new Dictionary<string, List<int>>();
            Operations = new Dictionary<string, List<long>>();

            var algorithms = new[] { "kruskal", "prim", "prim_heap",
                                   "theoretical_kruskal", "theoretical_prim", "theoretical_prim_heap" };

            foreach (var algo in algorithms)
            {
                Dimensions[algo] = new List<int>();
                Operations[algo] = new List<long>();
            }
        }

        /// <summary>
        /// Запустить эксперимент с размерностями от minSize до maxSize
        /// </summary>
        public void RunExperiment(int minSize, int maxSize, int step,
                                 string generatorType = "complete", int numTrials = 3)
        {
            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("ЭКСПЕРИМЕНТ: Исследование сложности алгоритмов МОД");
            Console.WriteLine(new string('=', 70));
            Console.WriteLine($"Тип графа: {generatorType}");
            Console.WriteLine($"Диапазон размеров: {minSize} - {maxSize}");
            Console.WriteLine($"Шаг: {step}");
            Console.WriteLine($"Повторений на размер: {numTrials}");
            Console.WriteLine(new string('=', 70) + "\n");

            for (int size = minSize; size <= maxSize; size += step)
            {
                Console.Write($"Размер графа: {size} вершин ... ");

                // Генерируем графы
                var graphs = new List<Graph>();
                for (int trial = 0; trial < numTrials; trial++)
                {
                    Graph graph;
                    int seed = size * 100 + trial;

                    if (generatorType == "complete")
                        graph = DataGenerator.GenerateCompleteGraph(size, 100.0, seed);
                    else if (generatorType == "sparse")
                        graph = DataGenerator.GenerateSparseGraph(size, 0.3, 100.0, seed);
                    else
                        graph = DataGenerator.GenerateEuclideanGraph(size, 100.0, seed);

                    graphs.Add(graph);
                }

                // Запускаем алгоритмы и собираем результаты
                var kruskalOps = new List<long>();
                var primOps = new List<long>();
                var primHeapOps = new List<long>();

                foreach (var graph in graphs)
                {
                    // Крускал
                    var kruskal = new KruskalAlgorithm();
                    kruskal.FindMST(graph);
                    kruskalOps.Add(kruskal.Operations);

                    // Прим
                    var prim = new PrimAlgorithm();
                    prim.FindMST(graph);
                    primOps.Add(prim.Operations);

                    // Прим с кучей
                    var primHeap = new PrimAlgorithmHeap();
                    primHeap.FindMST(graph);
                    primHeapOps.Add(primHeap.Operations);
                }

                // Берем среднее значение
                long avgKruskal = (long)kruskalOps.Average();
                long avgPrim = (long)primOps.Average();
                long avgPrimHeap = (long)primHeapOps.Average();

                // Сохраняем результаты
                Dimensions["kruskal"].Add(size);
                Operations["kruskal"].Add(avgKruskal);

                Dimensions["prim"].Add(size);
                Operations["prim"].Add(avgPrim);

                Dimensions["prim_heap"].Add(size);
                Operations["prim_heap"].Add(avgPrimHeap);

                // Теоретические значения
                int edgeCount = graphs[0].Edges.Count;

                long theoreticalK = (long)ComplexityAnalyzer.TheoreticalKruskal(size, edgeCount);
                long theoreticalP = (long)ComplexityAnalyzer.TheoreticalPrimSimple(size);
                long theoreticalPH = (long)ComplexityAnalyzer.TheoreticalPrimHeap(size, edgeCount);

                Dimensions["theoretical_kruskal"].Add(size);
                Operations["theoretical_kruskal"].Add(theoreticalK);

                Dimensions["theoretical_prim"].Add(size);
                Operations["theoretical_prim"].Add(theoreticalP);

                Dimensions["theoretical_prim_heap"].Add(size);
                Operations["theoretical_prim_heap"].Add(theoreticalPH);

                Console.WriteLine($"OK (Крускал: {avgKruskal}, Прим: {avgPrim}, Прим+куча: {avgPrimHeap})");
            }

            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("Эксперимент завершен!");
            Console.WriteLine(new string('=', 70) + "\n");
        }

        /// <summary>
        /// Вывести результаты в виде таблицы
        /// </summary>
        public void PrintResultsTable()
        {
            Console.WriteLine("\n" + new string('=', 100));
            Console.WriteLine("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ: Размерность - Количество операций");
            Console.WriteLine(new string('=', 100));
            Console.WriteLine($"{"Размер",8} | {"Крускал",12} | {"Прим",12} | {"Прим+куча",12} | " +
                            $"{"Т.Крускал",12} | {"Т.Прим",12} | {"Т.Прим+куча",12}");
            Console.WriteLine(new string('-', 100));

            for (int i = 0; i < Dimensions["kruskal"].Count; i++)
            {
                int size = Dimensions["kruskal"][i];
                long kExp = Operations["kruskal"][i];
                long pExp = Operations["prim"][i];
                long phExp = Operations["prim_heap"][i];
                long kTh = Operations["theoretical_kruskal"][i];
                long pTh = Operations["theoretical_prim"][i];
                long phTh = Operations["theoretical_prim_heap"][i];

                Console.WriteLine($"{size,8} | {kExp,12} | {pExp,12} | {phExp,12} | " +
                                $"{kTh,12} | {pTh,12} | {phTh,12}");
            }

            Console.WriteLine(new string('=', 100) + "\n");
        }

        /// <summary>
        /// Сохранить результаты в CSV файл
        /// </summary>
        public void SaveToCSV(string filename)
        {
            using (var writer = new StreamWriter(filename))
            {
                // Заголовок
                writer.WriteLine("Размер,Крускал,Прим,Прим+куча,Т.Крускал,Т.Прим,Т.Прим+куча");

                // Данные
                for (int i = 0; i < Dimensions["kruskal"].Count; i++)
                {
                    int size = Dimensions["kruskal"][i];
                    long kExp = Operations["kruskal"][i];
                    long pExp = Operations["prim"][i];
                    long phExp = Operations["prim_heap"][i];
                    long kTh = Operations["theoretical_kruskal"][i];
                    long pTh = Operations["theoretical_prim"][i];
                    long phTh = Operations["theoretical_prim_heap"][i];

                    writer.WriteLine($"{size},{kExp},{pExp},{phExp},{kTh},{pTh},{phTh}");
                }
            }

            Console.WriteLine($"Результаты сохранены в файл: {filename}");
        }
    }

    // ЧАСТЬ 6: ГЛАВНАЯ ПРОГРАММА
    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            // Демонстрация на маленьком графе
            DemoSmallExample();

            // Основной эксперимент
            RunFullAnalysis();

            Console.WriteLine("\nНажмите любую клавишу для завершения...");
            Console.ReadKey();
        }

        static void DemoSmallExample()
        {
            Console.WriteLine("\n" + new string('#', 70));
            Console.WriteLine("ДЕМОНСТРАЦИОННЫЙ ПРИМЕР: Минимальный остов на маленьком графе");
            Console.WriteLine(new string('#', 70));

            // Создаем пример графа
            var graph = new Graph(5);
            var edgesData = new (int, int, double)[]
            {
                (0, 1, 4), (0, 2, 2),
                (1, 2, 1), (1, 3, 5),
                (2, 3, 8), (2, 4, 10),
                (3, 4, 2), (1, 3, 7)
            };

            foreach (var (u, v, w) in edgesData)
            {
                graph.AddEdge(u, v, w);
            }

            Console.WriteLine($"\nГраф: {graph.VertexCount} вершин, {graph.Edges.Count} ребер");

            var sortedEdges = graph.GetEdgesSortedByWeight();
            Console.WriteLine("Ребра (отсортированные по весу):");
            foreach (var edge in sortedEdges)
            {
                Console.WriteLine($"  {edge}");
            }

            // Крускал
            Console.WriteLine("\n--- Алгоритм Крускала ---");
            var kruskal = new KruskalAlgorithm();
            var (kMSTEdges, kWeight) = kruskal.FindMST(graph);
            Console.WriteLine($"Ребра МОД: {string.Join(", ", kMSTEdges.Select(e => $"({e.U}-{e.V}:{e.Weight:F1})"))}");
            Console.WriteLine($"Вес МОД: {kWeight:F2}");
            Console.WriteLine($"Операций выполнено: {kruskal.Operations}");

            // Прим
            Console.WriteLine("\n--- Алгоритм Прима (простой) ---");
            var prim = new PrimAlgorithm();
            var (pMSTEdges, pWeight) = prim.FindMST(graph);
            Console.WriteLine($"Ребра МОД: {string.Join(", ", pMSTEdges.Select(e => $"({e.U}-{e.V}:{e.Weight:F1})"))}");
            Console.WriteLine($"Вес МОД: {pWeight:F2}");
            Console.WriteLine($"Операций выполнено: {prim.Operations}");

            // Прим с кучей
            Console.WriteLine("\n--- Алгоритм Прима (с кучей) ---");
            var primHeap = new PrimAlgorithmHeap();
            var (phMSTEdges, phWeight) = primHeap.FindMST(graph);
            Console.WriteLine($"Ребра МОД: {string.Join(", ", phMSTEdges.Select(e => $"({e.U}-{e.V}:{e.Weight:F1})"))}");
            Console.WriteLine($"Вес МОД: {phWeight:F2}");
            Console.WriteLine($"Операций выполнено: {primHeap.Operations}");
        }

        static void RunFullAnalysis()
        {
            Console.WriteLine("\n" + new string('*', 70));
            Console.WriteLine("*" + " АНАЛИЗ ВЫЧИСЛИТЕЛЬНОЙ ЭФФЕКТИВНОСТИ АЛГОРИТМОВ МОД ".PadRight(68) + "*");
            Console.WriteLine(new string('*', 70));

            var experiment = new PerformanceExperiment();

            // Запускаем эксперимент на полном графе
            experiment.RunExperiment(
                minSize: 5,
                maxSize: 30,
                step: 5,
                generatorType: "complete",
                numTrials: 3
            );

            // Выводим результаты в виде таблицы
            experiment.PrintResultsTable();

            // Сохраняем в CSV
            experiment.SaveToCSV("mst_results.csv");

            // Анализ
            Console.WriteLine(new string('=', 70));
            Console.WriteLine("АНАЛИЗ РЕЗУЛЬТАТОВ");
            Console.WriteLine(new string('=', 70));

            var dims = experiment.Dimensions["kruskal"];
            var kOps = experiment.Operations["kruskal"];
            var pOps = experiment.Operations["prim"];
            var phOps = experiment.Operations["prim_heap"];

            if (dims.Count >= 2)
            {
                double kGrowth = (double)kOps[kOps.Count - 1] / kOps[0];
                double pGrowth = (double)pOps[pOps.Count - 1] / pOps[0];
                double phGrowth = (double)phOps[phOps.Count - 1] / phOps[0];
                double sizeGrowth = (double)dims[dims.Count - 1] / dims[0];

                Console.WriteLine($"\nРост размера: {dims[0]} -> {dims[dims.Count - 1]} ({sizeGrowth:F1}x)");
                Console.WriteLine($"\nРост операций:");
                Console.WriteLine($"  Крускал:          {kGrowth:F1}x (ожидается ~{sizeGrowth * sizeGrowth * Math.Log(sizeGrowth):F1}x для O(E log E))");
                Console.WriteLine($"  Прим (простой):   {pGrowth:F1}x (ожидается ~{sizeGrowth * sizeGrowth:F1}x для O(V²))");
                Console.WriteLine($"  Прим (с кучей):   {phGrowth:F1}x (ожидается ~{sizeGrowth * Math.Log(sizeGrowth):F1}x для O((V+E)log V))");

                // Определяем самый эффективный
                long finalBestVal = Math.Min(kOps[kOps.Count - 1], Math.Min(pOps[pOps.Count - 1], phOps[phOps.Count - 1]));
                string bestAlgo;

                if (kOps[kOps.Count - 1] == finalBestVal)
                    bestAlgo = "Крускал";
                else if (pOps[pOps.Count - 1] == finalBestVal)
                    bestAlgo = "Прим (простой)";
                else
                    bestAlgo = "Прим (с кучей)";

                Console.WriteLine($"\nНа графе с {dims[dims.Count - 1]} вершинами наиболее эффективен: {bestAlgo}");
                Console.WriteLine($"  Крускал:          {kOps[kOps.Count - 1]} операций");
                Console.WriteLine($"  Прим (простой):   {pOps[pOps.Count - 1]} операций");
                Console.WriteLine($"  Прим (с кучей):   {phOps[phOps.Count - 1]} операций");
            }

            Console.WriteLine(new string('=', 70) + "\n");
        }
    }
}
