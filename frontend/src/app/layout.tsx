import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "IndoStockSense Dashboard",
  description: "AI Market Sentiment Analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
