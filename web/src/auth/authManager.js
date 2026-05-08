export class AuthManager {
    static getToken() {
        return localStorage.getItem('jwt_token') || sessionStorage.getItem('jwt_token');
    }
    
    static setToken(token, remember = true) {
        if (remember) {
            localStorage.setItem('jwt_token', token);
        } else {
            sessionStorage.setItem('jwt_token', token);
        }
    }
    
    static clearToken() {
        localStorage.removeItem('jwt_token');
        sessionStorage.removeItem('jwt_token');
    }
    
    static isAuthenticated() {
        return !!this.getToken();
    }
    
    static getAuthHeader() {
        const token = this.getToken();
        return token ? `Bearer ${token}` : null;
    }
}
