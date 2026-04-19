import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface AuditResult {
  status: string;
  results: any;
}

@Injectable({
  providedIn: 'root'
})
export class AuditService {
  private apiUrl = 'http://127.0.0.1:8000/api/audit';

  constructor(private http: HttpClient) { }

  runAudit(file: File): Observable<AuditResult> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<AuditResult>(this.apiUrl, formData);
  }
}
