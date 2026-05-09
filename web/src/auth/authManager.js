export class AuthManager {
    static getToken() {
        return localStorage.getItem('access_token') || sessionStorage.getItem('access_token');
    }
    
    static setToken(token, remember = true) {
        if (remember) {
            localStorage.setItem('access_token', token);
        } else {
            sessionStorage.setItem('access_token', token);
        }
    }
    
    static clearToken() {
        localStorage.removeItem('access_token');
        sessionStorage.removeItem('access_token');
    }
    
    static isAuthenticated() {
        return !!this.getToken();
    }
    
    static getAuthHeader() {
        const token = this.getToken();
        return token ? `Bearer ${token}` : null;
    }
}
