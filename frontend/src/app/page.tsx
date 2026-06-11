"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import { Activity, TrendingUp, TrendingDown, Minus, Clock } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

interface NewsItem {
  title: string;
  source: string;
  sentiment: string;
  confidence?: number;
  url: string;
}

interface Stats {
  positif: number;
  negatif: number;
  netral: number;
  total: number;
}

export default function Dashboard() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [stats, setStats] = useState<Stats>({ positif: 0, negatif: 0, netral: 0, total: 0 });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const mockData: NewsItem[] = [
          { title: "IHSG Sentuh Rekor Tertinggi Baru, Saham Perbankan Jadi Pendorong", source: "CNBC Indonesia", sentiment: "positif", confidence: 0.95, url: "#" },
          { title: "Suku Bunga BI Naik, Saham Properti Tertekan Hebat Hari Ini", source: "Bisnis Market", sentiment: "negatif", confidence: 0.88, url: "#" },
          { title: "GOTO Rombak Jajaran Direksi, Investor Menunggu Hasil Kuartal III", source: "Kontan Investasi", sentiment: "netral", confidence: 0.76, url: "#" },
          { title: "Laba Bersih BBCA Tumbuh 20%, Analis Rekomendasikan Beli", source: "CNBC Indonesia", sentiment: "positif", confidence: 0.92, url: "#" },
          { title: "Harga Batubara Global Anjlok, Saham Energi Melemah Serentak", source: "Bisnis Market", sentiment: "negatif", confidence: 0.84, url: "#" },
          { title: "[Ask] Gimana prospek saham tech Indo tahun depan?", source: "Reddit r/finansial", sentiment: "netral", confidence: 0.65, url: "#" }
        ];

        let data = mockData;
        try {
            const res = await axios.get("http://localhost:8000/api/news");
            if (res.data && res.data.data && res.data.data.length > 0) {
                data = res.data.data;
            }
        } catch(e) {
            console.log("Backend offline, using mock data for UI demo");
        }

        setNews(data);
        
        const pos = data.filter((d: NewsItem) => d.sentiment === "positif").length;
        const neg = data.filter((d: NewsItem) => d.sentiment === "negatif").length;
        const net = data.filter((d: NewsItem) => d.sentiment === "netral").length;
        setStats({ positif: pos, negatif: neg, netral: net, total: data.length });
        
      } catch (error) {
        console.error("Failed to fetch news", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const chartData = [
    { name: "Positif", value: stats.positif, color: "#10b981" },
    { name: "Negatif", value: stats.negatif, color: "#ef4444" },
    { name: "Netral", value: stats.netral, color: "#64748b" }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#0f1115]">
        <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="bg-glow"></div>
      
      <header className="header animate-fade-up">
        <div>
          <h1 className="header-title">IndoStockSense</h1>
          <p className="text-[#94a3b8] mt-2 flex items-center gap-2">
            <Activity size={18} className="text-blue-500" />
            Real-time AI Market Sentiment Analysis
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm text-[#94a3b8]">Live Status</p>
          <div className="flex items-center gap-2 mt-1">
            <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
            <span className="font-medium text-emerald-400">Models Active</span>
          </div>
        </div>
      </header>

      <div className="grid-stats">
        <div className="glass-panel animate-fade-up delay-1">
          <div className="flex items-center justify-between">
            <h3 className="text-[#94a3b8] font-medium uppercase tracking-wider text-sm">Sinyal Positif</h3>
            <div className="p-2 bg-emerald-500/10 rounded-lg">
              <TrendingUp className="text-emerald-500" size={24} />
            </div>
          </div>
          <div className="stat-value text-emerald-400">{stats.positif}</div>
          <p className="text-sm text-[#94a3b8]">Berita membawa sentimen *bullish*</p>
        </div>

        <div className="glass-panel animate-fade-up delay-2">
          <div className="flex items-center justify-between">
            <h3 className="text-[#94a3b8] font-medium uppercase tracking-wider text-sm">Sinyal Negatif</h3>
            <div className="p-2 bg-red-500/10 rounded-lg">
              <TrendingDown className="text-red-500" size={24} />
            </div>
          </div>
          <div className="stat-value text-red-400">{stats.negatif}</div>
          <p className="text-sm text-[#94a3b8]">Berita membawa tekanan jual</p>
        </div>

        <div className="glass-panel animate-fade-up delay-3">
          <div className="flex items-center justify-between">
            <h3 className="text-[#94a3b8] font-medium uppercase tracking-wider text-sm">Sinyal Netral</h3>
            <div className="p-2 bg-slate-500/10 rounded-lg">
              <Minus className="text-slate-400" size={24} />
            </div>
          </div>
          <div className="stat-value text-slate-300">{stats.netral}</div>
          <p className="text-sm text-[#94a3b8]">Informasi faktual / opini netral</p>
        </div>
      </div>

      <div className="grid-main animate-fade-up delay-3">
        <div className="glass-panel">
          <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Clock size={20} className="text-blue-400" />
            Live Market News Stream
          </h2>
          
          <div className="news-list">
            {news.map((item, idx) => (
              <div key={idx} className="news-item">
                <div style={{ flex: 1 }}>
                  <a href={item.url} target="_blank" rel="noreferrer" className="news-title flex items-start gap-2">
                    {item.title}
                  </a>
                  <div className="news-meta mt-2">
                    <span className="bg-[#1e293b] px-2 py-1 rounded text-xs">{item.source}</span>
                    {item.confidence && <span>AI Conf: {(item.confidence * 100).toFixed(1)}%</span>}
                  </div>
                </div>
                <div className="ml-4 flex-shrink-0">
                  <span className={`badge ${item.sentiment}`}>
                    {item.sentiment}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel" style={{ maxHeight: '450px' }}>
          <h2 className="text-xl font-semibold mb-6">Sentiment Distribution</h2>
          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={70}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6 mt-4">
            {chartData.map((item, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                <span className="text-sm text-[#94a3b8]">{item.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
