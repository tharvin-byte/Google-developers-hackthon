import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AuditService, AuditResult } from './audit.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class AppComponent {
  title = 'FairSight';
  file = signal<File | null>(null);
  loading = signal(false);
  result = signal<AuditResult | null>(null);
  error = signal<string | null>(null);

  constructor(private auditService: AuditService) {}

  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    if (file) {
      this.file.set(file);
    }
  }

  runAudit() {
    const currentFile = this.file();
    if (!currentFile) return;

    this.loading.set(true);
    this.error.set(null);
    this.result.set(null);

    this.auditService.runAudit(currentFile).subscribe({
      next: (res) => {
        this.result.set(res);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err.error?.detail || 'An unexpected error occurred.');
        this.loading.set(false);
      }
    });
  }

  getVerdictClass(verdict: string): string {
    if (verdict === 'FAIR') return 'status-fair';
    if (verdict === 'MODERATE') return 'status-moderate';
    return 'status-biased';
  }

  getScoreColor(score: number): string {
    if (score >= 0.8) return 'var(--fair-color)';
    if (score >= 0.5) return 'var(--moderate-color)';
    return 'var(--biased-color)';
  }
}
